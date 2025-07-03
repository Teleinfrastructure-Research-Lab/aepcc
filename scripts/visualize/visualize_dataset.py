import open3d as o3d
from tqdm import tqdm
import sys
import traceback

from utils.utils import parse_kwargs, onehot_to_str
from utils.visualize import create_xy_grid
from smol.core import smol
from smol.register.DataLoader import CommonDataLoader


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    smol.register()

    if len(sys.argv) <2:
        smol.logger.error("Please provide arguments!")
        smol.logger.debug(traceback.format_exc())
        sys.exit(1)
    else:
        try:
            dl_name = sys.argv[1]
            dl_params = parse_kwargs(sys.argv[2:])
        except Exception as e:
            smol.logger.error(f"Exception while parsing kwargs: {e}")
            smol.logger.debug(traceback.format_exc())
            sys.exit(1)
    
    dl = CommonDataLoader(dl_name, dl_params)

    print(f"Total of {len(dl)} batches prepared for joyful loading!")

    for batch in tqdm(dl):
        point_clouds_tensor = batch['point_clouds']
        categories = onehot_to_str(batch['cls_onehots'])
        paths = batch['paths']

        pc_np = point_clouds_tensor.numpy()

        for i in range(len(pc_np)):
            print(f"Visualizing element {i}(Category: {categories[i]})")

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc_np[i].T)

            # Create grid
            grid_size = 1.0  # Adjust the size of the grid
            grid_divisions = 10  # Adjust the number of divisions in the grid
            grid_lines = create_xy_grid(size=grid_size, n=grid_divisions)

            # Visualize the point clouds and grid
            print(f"Drawing mesmerizing point clouds ...")
            print(f"{paths[i]}")
            o3d.visualization.draw_geometries([pcd] + grid_lines)

            # Wait for user input to continue to the next point cloud
            input("Press any key to continue...")
