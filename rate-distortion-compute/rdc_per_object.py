import os
import click
import torch
import pandas as pd
from tqdm import tqdm
import time


from smol.core import smol
from smol.register.Architecture import CommonArchitecture
from smol.register.DataLoader import CommonDataLoader
from smol.register.Loss import CommonLoss

from utils.codec import bounding_box_diagonal_masked, compress_object, remove_key_from_signal
from utils.utils import d1_psnr_pcc, onehot_to_str


REMOVE_IDX_FROM_IMR = True



@click.command()
@click.argument('input_path', type=click.Path(exists=True))
def rdc_per_object(input_path):
    """
    If input_path is a file, process that single YAML.
    Collect results into a single CSV, which includes an additional 'arch_name' column.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smol.register()

    if os.path.isfile(input_path):
        config_files = [input_path]
        out_basename = os.path.splitext(os.path.basename(input_path))[0]
    else:
        raise IsADirectoryError()

    results_df = pd.DataFrame(columns=["rate", "distortion", "arch_name", "category"])
    smol.add_runtime_configs(config_files)

    for config_file in config_files:

        exp_name = os.path.splitext(os.path.basename(config_file))[0]
        try: 
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
            if dl.config["batch_size"] != 1:
                raise ValueError("rdc: batch_size must be 1")

            for batch in tqdm(dl, desc=f"Processing {exp_name}", total=len(dl)):
                with torch.no_grad():
                    batch.to(device)
                    category = onehot_to_str(batch["cls_onehots"])[0]
                    # ----- measure inference time -----
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    out = arch(batch)  # <---- timed

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    infer_time_s = t1 - t0
                    # --------------------------------------------------
                    x = batch["point_clouds"].squeeze(0)
                    mask = batch["masks"].squeeze(0)
                    ref = x[:, mask==1.0]
                    test = out["out"].squeeze(0)
                    if "gl_idx" in out:
                        test = test[:, mask==1.0]
                    psnr_value = d1_psnr_pcc(ref.unsqueeze(0), test.unsqueeze(0), chunk=2048, device='cuda' if torch.cuda.is_available() else 'cpu')

                    if "gl_idx" in out and REMOVE_IDX_FROM_IMR == True:
                        bits = len(compress_object(remove_key_from_signal(remove_key_from_signal(out, "out"), "gl_idx"))) * 8
                    else:
                        bits = len(compress_object(remove_key_from_signal(out, "out"))) * 8

                    points = mask.sum()
                    bpp = bits / points

                    results_df = pd.concat([
                        results_df,
                        pd.DataFrame([{
                            "rate": bpp.item(),
                            "distortion": psnr_value,
                            "arch_name": arch_name,
                            "category": category,
                            "inference_time_s": infer_time_s,
                        }])
                    ], ignore_index=True)
        except Exception as e:
            smol.logger.error(f"rdc evaluation of {exp_name} failed: {e}")
    
    os.makedirs("outputs", exist_ok=True)
    out_csv_path = os.path.join("outputs", "rdc_outputs", f"{out_basename}.csv")
    results_df.to_csv(out_csv_path, index=False)

if __name__ == "__main__":
    rdc_per_object()
