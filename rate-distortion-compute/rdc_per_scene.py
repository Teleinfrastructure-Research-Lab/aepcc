import os
import time
import json
import torch
import click
import pandas as pd
import numpy as np
import open3d as o3d
from tqdm import tqdm
import traceback

from smol.core import smol
from smol.signal import Signal
from smol.register.Architecture import CommonArchitecture

from utils.codec import compress_scene, remove_key_from_signal, remove_isolated_points
from utils.utils import d1_psnr_pcc
from utils.DataPipeline.h5 import sample_mt_snn, zero_fill_torch
from utils.geometry import sphere_normalization_with_params, sphere_denormalization



REMOVE_IDX_FROM_IMR = True



def get_synonym(df: pd.DataFrame, search_string: str) -> str:
    row = df[df['original'] == search_string]
    return row['synonym'].values[0] if not row.empty else None

def create_point_cloud(points: torch.Tensor, mask: torch.Tensor = None) -> o3d.geometry.PointCloud:
    if points.shape[0] != 1 or points.shape[1] != 3:
        raise ValueError("Expected points of shape (1, 3, N)")

    points_flat = points.squeeze(0).permute(1, 0)
    if mask is not None:
        mask_flat = mask.squeeze(0).bool()
        points_flat = points_flat[mask_flat]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_flat.detach().cpu().numpy())
    return point_cloud

def load_scene_objects(scene, merged_df):
    return merged_df[
        (merged_df["scene_id"] == scene["name"]) &
        (merged_df["dataset"] == scene["dataset"]) &
        (merged_df["region"] == scene["region"])
    ]

def prepare_input_signal(obj, num_points, use_synonyms, synonyms, device):
    path = obj["path"]
    category = obj["category"]
    category_synonym = get_synonym(synonyms, category)
    if use_synonyms and category_synonym is None:
        return None

    vertices = json.loads(obj["vertices"])
    points = sample_mt_snn(path, vertices, num_points)
    sample_points_num = points.shape[1]

    points_pt = torch.tensor(points, dtype=torch.float32)
    points_pt, centroid, max_distance = sphere_normalization_with_params(points_pt.T)
    points_pt, mask_pt = zero_fill_torch(points_pt.T, num_points, sample_points_num) if sample_points_num < num_points else (points_pt.T, torch.ones((1, num_points)))
    points_pt = points_pt.unsqueeze(0)

    cls_dim = smol.get_config("data", "CLASS_NUM")
    cls_onehot = torch.zeros((1, cls_dim, 1))

    signal = Signal({
        "point_clouds": points_pt,
        "cls_onehots": cls_onehot,
        "masks": mask_pt
    }, {
        "point_clouds": (-1, 3, -1),
        "cls_onehots": (-1, cls_dim, 1),
        "masks": (-1, -1)
    }).to(device)

    return signal, centroid, max_distance, points_pt, mask_pt

def evaluate_scene(scene_objects, arch, num_points, use_synonyms, synonyms, device):
    pc_in, pc_out, mask, imr_data = None, None, None, []
    valid_count = 0

    num_input_points  = 0
    num_output_points = 0

    for _, obj in tqdm(scene_objects.iterrows(), total=len(scene_objects)):
        result = prepare_input_signal(obj, num_points, use_synonyms, synonyms, device)
        if result is None:
            continue

        try:
            signal, centroid, max_distance, input_pts, mask_pt = result
            with torch.no_grad():
                out = arch(signal).to('cpu')

            obj_input  = sphere_denormalization(input_pts.squeeze(0).T, centroid, max_distance).T.unsqueeze(0)
            obj_output = sphere_denormalization(out["out"].squeeze(0).T, centroid, max_distance).T.unsqueeze(0)
            if "gl_idx" in out:
                obj_output = remove_isolated_points(obj_output, out["gl_idx"])

            pc_in  = obj_input  if pc_in  is None else torch.cat((pc_in,  obj_input),  dim=2)
            pc_out = obj_output if pc_out is None else torch.cat((pc_out, obj_output), dim=2)
            mask   = mask_pt    if mask   is None else torch.cat((mask,   mask_pt),   dim=1)

            if "gl_idx" in out and REMOVE_IDX_FROM_IMR:
                imr = Signal({"centroid": centroid, "max_distance": max_distance},
                             {"centroid": (-1, 3), "max_distance": ()}
                            ).join(remove_key_from_signal(remove_key_from_signal(out, "out"), "gl_idx"))
            else:
                imr = Signal({"centroid": centroid, "max_distance": max_distance},
                             {"centroid": (-1, 3), "max_distance": ()}
                            ).join(remove_key_from_signal(out, "out"))
            imr_data.append(imr)

            num_input_points  += int(mask_pt.sum().item())
            num_output_points += obj_output.shape[2]

            valid_count += 1
            del out, imr, signal, input_pts, mask_pt
            torch.cuda.empty_cache()
        except Exception as e:
            smol.logger.error(f"Error while processing object: {obj['id']}: {e}")
            smol.logger.error(traceback.format_exc())

    if valid_count == 0:
        return None, None, None, [], 0, 0, 0

    return pc_in, pc_out, mask, imr_data, valid_count, num_input_points, num_output_points

def save_point_clouds(pc_in, pc_out, mask, basename, use_mask):
    pcd_in = create_point_cloud(pc_in, mask)
    pcd_out = create_point_cloud(pc_out, mask if use_mask else None)

    output_dir = os.path.join(smol.get_config("paths", "TEST_OUTPUTS_DIR"), "rdc_outputs", "scenes")
    os.makedirs(output_dir, exist_ok=True)
    in_path = os.path.join(output_dir, f"{basename}_in.ply")
    out_path = os.path.join(output_dir, f"{basename}_out.ply")

    o3d.io.write_point_cloud(in_path, pcd_in)
    o3d.io.write_point_cloud(out_path, pcd_out)
    del pcd_in, pcd_out

def process_config_file(config_file, merged_df, synonyms, results_df, out_basename, device):
    exp_name = os.path.splitext(os.path.basename(config_file))[0]
    scenes = smol.get_config(exp_name, "scenes")
    use_synonyms = smol.get_config(exp_name, "use_synonyms")
    mask_output = smol.get_config(exp_name, "mask_output")

    arch_name = smol.get_config(exp_name, "architecture", "name")
    arch_params = smol.get_config(exp_name, "architecture", "module_params")
    arch_ckpt = smol.get_config(exp_name, "architecture", "load_ckpt")

    arch = CommonArchitecture(arch_name, arch_params).to(device)
    arch.load_checkpoints(*arch_ckpt.values())
    arch.eval()

    num_points = smol.get_config("data", "NUM_POINTS")

    for scene in scenes:
        try:
            smol.logger.info(f"Processing scene {scene['name']}:region{scene['region']} from {scene['dataset']}")
            scene_objects = load_scene_objects(scene, merged_df)
            t0 = time.perf_counter()
            pc_in, pc_out, mask, imrs, count, num_input_points, num_output_points = evaluate_scene(scene_objects, arch, num_points, use_synonyms, synonyms, device)
            eval_time_s = time.perf_counter() - t0
            if count == 0:
                smol.logger.warning(f"No valid objects found for scene {scene['name']} – skipping.")
                continue

            psnr = d1_psnr_pcc(pc_in[:,:,mask[0,:]==1.0], pc_out, chunk=2048, device='cuda' if torch.cuda.is_available() else 'cpu')
            bits = len(compress_scene(imrs)) * 8
            bpp = bits / mask.sum()

            scene_id = f"{out_basename}_{scene['dataset']}_{scene['name']}_{scene['region']}"

            results_df = pd.concat([results_df, pd.DataFrame([{ 
                "rate": bpp.item(),
                "num_objects": count,
                "num_input_points": num_input_points,
                "num_output_points": num_output_points,
                "distortion": psnr,
                "arch_name": arch_name,
                "scene": scene_id,
                "use_synonyms": str(use_synonyms),
                "evaluate_time_s": eval_time_s,
            }])], ignore_index=True)

            # save_point_clouds(pc_in, pc_out, mask, f"{scene_id}_{use_synonyms}", mask_output)
            del pc_in, pc_out, mask, imrs
            torch.cuda.empty_cache()
        except Exception as e:
            smol.logger.error(f"Error while processing scene: {scene['name']}:region{scene['region']} from {scene['dataset']}: {e}")
            smol.logger.error(traceback.format_exc())

    return results_df

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
def rdc_per_scene(input_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smol.register()

    if os.path.isfile(input_path):
        config_files = [input_path]
        out_basename = os.path.splitext(os.path.basename(input_path))[0]
    else:
        raise IsADirectoryError("Input path must be a file.")

    smol.add_runtime_configs(config_files)
    exp_name = os.path.splitext(os.path.basename(config_files[0]))[0]
    use_synonyms = smol.get_config(exp_name, "use_synonyms")

    mt_samples = pd.read_csv(smol.get_config("data", "MT_SAMPLES_PATH"), sep=";")
    mt_samples['region'] = mt_samples['path'].str.extract(r'region(\d+)\.ply').astype(int)
    snn_samples = pd.read_csv(smol.get_config("data", "SNN_SAMPLES_PATH"), sep=";")
    snn_samples['region'] = 0
    synonyms = pd.read_csv(smol.get_config("data", "SYNONYMS_PATH"), sep=";")

    merged_df = pd.concat([mt_samples, snn_samples], ignore_index=True)
    results_df = pd.DataFrame(columns=[
        "rate",
        "num_objects",
        "num_input_points",
        "num_output_points",
        "distortion",
        "arch_name",
        "scene",
        "use_synonyms",
        "evaluate_time_s",
    ])

    for config_file in config_files:
        results_df = process_config_file(config_file, merged_df, synonyms, results_df, out_basename, device)

    output_path = os.path.join(smol.get_config("paths", "TEST_OUTPUTS_DIR"), "rdc_outputs", f"{out_basename}_{use_synonyms}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    rdc_per_scene()
