import h5py
import numpy as np
import torch
import json
import math

from smol.core import smol
from utils.utils import str_to_onehot


class HDF5Reader:
    """Efficient sequential reading from an HDF5 file without torch DataLoader."""

    def __init__(self, h5_path:str, split:str = 'train',
                 max_points:int = smol.get_config("data", "NUM_POINTS")):
        self.h5_path = h5_path
        self.split = split
        self.max_points = max_points

        # Define dataset names
        if split == "train":
            self.input_dataset_name = "train_input"
            self.label_dataset_name = "train_label"
        elif split == "val":
            self.input_dataset_name = "val"
            self.label_dataset_name = "val_label"
        elif split == "test":
            self.input_dataset_name = "test"
            self.label_dataset_name = "test_label"
        else:
            raise ValueError(f"There is no split: {split}")

        cache_size = smol.get_config("data", "CACHE_SIZE_MB") * 1024**2
        num_slots = smol.get_config("data", "CACHE_SLOTS")
        policy = smol.get_config("data", "CACHE_POLICY")

        self.h5_file = h5py.File(h5_path, 'r', 
                         rdcc_nbytes=cache_size,  # Cache size for raw data chunk cache
                         rdcc_w0=policy,            # Policy for preempting chunks (0 = LRU, 1 = newest)
                         rdcc_nslots=num_slots)   # Number of chunk slots in the hash table
        self.input_ds = self.h5_file[self.input_dataset_name]
        self.label_ds = self.h5_file[self.label_dataset_name]
        self.dataset_size = len(self.input_ds)

    def __len__(self):
        return self.dataset_size

    def get_batch(self, start_idx, batch_size):
        """Fetches a batch of data given a start index and batch size."""
        end_idx = min(start_idx + batch_size, self.dataset_size)

        # Direct batch slicing
        data_batch = self.input_ds[start_idx:end_idx]  
        label_batch = self.label_ds[start_idx:end_idx]  # This is still JSON

        batch_points = []
        batch_masks = []
        batch_categories = []
        batch_paths = []

        for data, label in zip(data_batch, label_batch):
            mask = data[-1, :]
            points = data[:-1, :]

            points, mask = self.subsample_points(points, mask, self.max_points)

            batch_points.append(torch.Tensor(points))
            batch_masks.append(torch.Tensor(mask).int().bool())
            batch_categories.append(json.loads(label)["category"])
            batch_paths.append(json.loads(label)["id"])

        return {
            'point_clouds': torch.stack(batch_points),
            'cls_onehots': str_to_onehot(batch_categories),
            'masks': torch.stack(batch_masks),
            'paths': batch_paths
        }

    def subsample_points(self, points: np.ndarray, mask: np.ndarray, num_samples: int):
        """ Subsamples points based on a mask while maintaining balance """
        ones_indices = np.where(mask == 1)[0]
        zeros_indices = np.where(mask == 0)[0]

        num_ones = len(ones_indices)
        if num_ones >= num_samples:
            selected_indices = np.random.choice(ones_indices, num_samples, replace=False)
        else:
            selected_indices = ones_indices.tolist()
            remaining_needed = num_samples - num_ones
            if remaining_needed > 0:
                additional_indices = zeros_indices[:remaining_needed]
                selected_indices.extend(additional_indices)

        return points[:, selected_indices], mask[selected_indices]

    def close(self):
        """Closes the HDF5 file."""
        self.h5_file.close()

class BatchGenerator:
    def __init__(self, generator_func, dataset_size, batch_size, num_batches):
        """
        Wraps a generator function to make it iterable and length-aware.
        """
        self.generator_func = generator_func  # Function that returns a generator
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_batches = num_batches

    def __iter__(self):
        """Returns a fresh generator every time it's iterated."""
        return self.generator_func()

    def __len__(self):
        """Returns the number of batches."""
        return self.num_batches

@smol.register_dl("default-dl", {
    "output_def": {
        "point_clouds":(-1, 3, smol.get_config("data","NUM_POINTS")),
        "cls_onehots":(-1, smol.get_config("data","CLASS_NUM"), 1),
        "masks":(-1, smol.get_config("data","NUM_POINTS")),
        "paths":(-1,),
    },
    "dl_params": {
        "h5_path": smol.get_config("data","H5_PATH_SYNTH"),
        "batch_size": 4,
        "split": "train",
        "max_points": smol.get_config("data","NUM_POINTS")
    }
})
def get_dataloader(h5_path,
                   batch_size=smol.get_config("default-params","training","BATCH_SIZE"),
                   split='train',
                   max_points=smol.get_config("data", "NUM_POINTS")):
    """
    Returns a length-aware generator that iterates over the dataset in batches,
    ensuring all batches are full. If the end is reached, it wraps around.
    """

    reader = HDF5Reader(h5_path, split, max_points)
    dataset_size = len(reader)
    num_batches = math.ceil(dataset_size / batch_size)

    def generator():
        index = 0
        while index < dataset_size:
            # Read batch directly from HDF5
            batch = reader.get_batch(index, batch_size)

            yield batch

            index += batch_size

    return BatchGenerator(generator, dataset_size, batch_size, num_batches)