import pandas as pd
import open3d as o3d
import numpy as np
import json
import torch
import random
import h5py
from tqdm import tqdm
from typing import Tuple

from utils.geometry import sphere_normalization, apply_random_rotation, add_random_noise, remove_points_fuzzy_sphere
from .MeshCache import MeshCache
from smol.core import smol

CACHING_ENABLED = True
CACHE_SIZE = 2000
if CACHING_ENABLED:
    mc = MeshCache(CACHE_SIZE)

def write_split(h5ds_points: h5py.Dataset, h5ds_labels: h5py.Dataset, df:pd.DataFrame, num_points:int, aug_std:float, aug_radius:float)->None:

    buf_size = smol.get_config("data", "H5_SAMPLE_DIM_CHUNK_SIZE")
    buff_data = np.zeros((buf_size, 4, num_points), dtype=np.float32)
    buff_labels = np.zeros((buf_size,), dtype=h5py.special_dtype(vlen=str))
    buff_len = 0

    for i, row in tqdm(df.iterrows(), total = len(df)):
        id = row["id"]
        path = row["path"]
        dataset = row["dataset"]
        category = row["category"]
        aug_flag = row["augmented"]
        vertices = []
        # READ ------------------------------------------------------------
        if dataset in ["mt", "snn"]:
            vertices = json.loads(row["vertices"])
            points = sample_mt_snn(path, vertices, num_points)
        else:
            points = sample_shapenet(path, num_points)
        # READ ------------------------------------------------------------

        # AUGMENT ---------------------------------------------------------
        if points.size == 0:
            smol.logger.error(f"{path}: Empty point cloud. Skiping!")
            continue
        points_pt = torch.Tensor(points)

        if aug_flag > 0:
            if dataset in ["mt", "snn"]:
                points_pt = apply_augmentations_and_normalize(points_pt.permute(1,0), aug_std, aug_radius, remove=False).permute(1,0)
            else:
                points_pt = apply_augmentations_and_normalize(points_pt.permute(1,0), aug_std, aug_radius).permute(1,0)
        else:
            points_pt = sphere_normalization(points_pt.permute(1,0)).permute(1,0)
        points = points_pt.numpy()
        check_array_stats(points, (-0.1, 0.1), (0, 1.1), (-1.1, 0))
        # AUGMENT ---------------------------------------------------------

        # WRITE -----------------------------------------------------------
        sample_points_num = points.shape[1]
        if sample_points_num < num_points:
            points, mask = zero_fill(points, num_points, sample_points_num)
        else:
            mask = np.ones((1, num_points))

        buff_data[buff_len] = np.vstack([points, mask])
        buff_labels[buff_len] = json.dumps({"id": id, "dataset": dataset, "category": category})
        buff_len += 1

        if buff_len == buf_size:
            start_idx = i - buff_len + 1
            end_idx = start_idx + buff_len
            try:
                h5ds_points[start_idx:end_idx] = buff_data[:buff_len]
                h5ds_labels[start_idx:end_idx] = buff_labels[:buff_len]
            except Exception as e:
                smol.logger.error(f"Exception when writing to h5: {e}")
            buff_len = 0
        # WRITE -----------------------------------------------------------
        
    # Write leftovers
    if buff_len > 0:
        start_idx = len(df) - buff_len
        end_idx = len(df)
        try:
            h5ds_points[start_idx:end_idx] = buff_data[:buff_len]
            h5ds_labels[start_idx:end_idx] = buff_labels[:buff_len]
        except Exception as e:
            smol.logger.error(f"Exception when writing to h5: {e}")



def zero_fill(points: np.array, _num_points:int, _sample_points_num:int) -> Tuple[np.array, np.array]:
    missing_points = _num_points - _sample_points_num
    zero_filled_points = np.zeros((3, missing_points))
    _points = np.hstack([points, zero_filled_points])
    mask = np.hstack([np.ones((1, _sample_points_num)), np.zeros((1, missing_points))])

    return _points, mask

def zero_fill_torch(points: torch.Tensor, num_points: int, sample_points_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
    missing_points = num_points - sample_points_num
    zero_filled_points = torch.zeros((3, missing_points), dtype=points.dtype, device=points.device)
    _points = torch.cat([points, zero_filled_points], dim=1)

    mask = torch.cat([
        torch.ones((1, sample_points_num), dtype=points.dtype, device=points.device),
        torch.zeros((1, missing_points), dtype=points.dtype, device=points.device)
    ], dim=1)

    return _points, mask

def load_mesh_cached(_path):
    if CACHING_ENABLED:
        return mc.load_mesh(_path)
    else:
        return o3d.io.read_triangle_mesh(_path)

def sample_mt_snn(_path:str, _vertices:list, _num_points:int) -> np.array:
    mesh = load_mesh_cached(_path)
    points = np.asarray(mesh.vertices)[_vertices]
    sample_points_num = points.shape[0]
    points = points.transpose()
    if sample_points_num > _num_points:
        selected_indices = np.random.choice(sample_points_num, size=_num_points, replace=False)
        points = points[:, selected_indices]
    return points

def sample_shapenet(_path:str, _num_points:int) -> np.array:
    mesh = load_mesh_cached(_path)
    point_cloud = mesh.sample_points_uniformly(number_of_points=_num_points)
    points = np.asarray(point_cloud.points).transpose()
    return points

def apply_augmentations_and_normalize(point_cloud:torch.Tensor, _std:float, _radius:float, rotate:bool = True, noise:bool = True, remove:bool = True) -> torch.Tensor:

    aug_pc = sphere_normalization(point_cloud)
    if rotate == True:
        aug_pc = apply_random_rotation(point_cloud)
    if random.random() < 0.5 and noise == True:
        aug_pc = add_random_noise(aug_pc, std=_std)
    aug_pc = sphere_normalization(aug_pc)
    if random.random() < 0.5 and remove == True:
        aug_pc = remove_points_fuzzy_sphere(aug_pc, radius=_radius)
    aug_pc = sphere_normalization(aug_pc)

    return aug_pc


def check_array_stats(arr: np.ndarray, mean_range: tuple, max_range: tuple, min_range: tuple):
    """
    Checks if the mean, std, max, and min of a NumPy array fall within specified intervals.
    Raises ValueError if any condition is not met.
    
    :param arr: NumPy array to check
    :param mean_range: Tuple (min, max) for allowed mean range
    :param std_range: Tuple (min, max) for allowed standard deviation range
    :param max_range: Tuple (min, max) for allowed max value range
    :param min_range: Tuple (min, max) for allowed min value range
    """
    mean_val = np.mean(arr)
    max_val = np.max(arr)
    min_val = np.min(arr)
    
    if not (mean_range[0] <= mean_val <= mean_range[1]):
        raise ValueError(f"Mean value {mean_val} is out of range {mean_range}")
    
    if not (max_range[0] <= max_val <= max_range[1]):
        raise ValueError(f"Max value {max_val} is out of range {max_range}")
    
    if not (min_range[0] <= min_val <= min_range[1]):
        raise ValueError(f"Min value {min_val} is out of range {min_range}")