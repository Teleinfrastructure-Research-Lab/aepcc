import torch
import open3d as o3d
import click
import os

from smol.core import smol
from smol.signal import Signal
from smol.register.Architecture import CommonArchitecture
from smol.register.DataLoader import CommonDataLoader


@click.command()
@click.argument('input_path', type=click.Path(exists=True, file_okay=True, dir_okay=False))
def visualize_exp_output(input_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smol.register()
    smol.add_runtime_configs([os.path.normpath(input_path)])
    exp_name = os.path.basename(input_path).split('.')[:-1][0]

    arch_name = smol.get_config(exp_name, "architecture", "name")
    arch_params = smol.get_config(exp_name, "architecture", "module_params")
    arch_load_ckpt = smol.get_config(exp_name, "architecture", "load_ckpt")
    arch = CommonArchitecture(arch_name, arch_params)
    arch.to(device)
    arch.load_checkpoints(*arch_load_ckpt.values())
    arch.eval()

    dl_name = smol.get_config(exp_name, "dataloader-test", "name")
    dl_params = smol.get_config(exp_name, "dataloader-test", "dl_params")
    dl = CommonDataLoader(dl_name, dl_params)

    for batch in dl:
        with torch.no_grad():
            batch_size = batch["point_clouds"].shape[0]
            batch.to(device)
            out = arch(batch)
            y = out["out"]
            for i in range(batch_size):
                pcs_np = batch["point_clouds"][i].detach().cpu().numpy()
                y_np = y[i].detach().cpu().numpy()
                mask_np = batch["masks"][i].detach().cpu().numpy()
                
                pcs_filtered = pcs_np[:, mask_np]
                y_filtered = y_np

                shift_amount = 3.0
                pcs_pcd = o3d.geometry.PointCloud()
                pcs_pcd.points = o3d.utility.Vector3dVector(pcs_filtered.T)
                pcs_pcd.paint_uniform_color([1, 0, 0])  # Input points in red

                y_pcd = o3d.geometry.PointCloud()
                y_pcd.points = o3d.utility.Vector3dVector(y_filtered.T)
                y_pcd.paint_uniform_color([0, 1, 0])  # Output points in green

                y_pcd.translate((shift_amount, 0, 0))  # Shift along x-axis

                # Display the point clouds
                o3d.visualization.draw_geometries([pcs_pcd, y_pcd])

                input("Press any key to continue...")



if __name__ =="__main__":
    visualize_exp_output()
    
