
import time
import h5py
import numpy as np
from tqdm import tqdm
import torch

from smol.core import smol

# Path to your HDF5 file
h5_path = smol.get_config("data","H5_PATH_SYNTH")
input_dataset_name = "train_input"  # Adjust based on your dataset
batch_size = 512  # Test different batch sizes

# Open the HDF5 file
with h5py.File(h5_path, 'r') as f:
    dataset = f[input_dataset_name]
    dataset_size = len(dataset)

    print(f"Dataset size: {dataset_size}")
    print(f"Reading {batch_size} samples at a time...")

    start_time = time.time()
    for i in tqdm(range(0, dataset_size, batch_size), total = dataset_size//batch_size):
        batch = dataset[i:i+batch_size]  # Read a batch sequentially
        a = torch.Tensor(np.array(batch))  # Convert to NumPy to force the read operation

    elapsed_time = time.time() - start_time
    print(f"Total read time: {elapsed_time:.4f} sec")
    print(f"Samples per second: {dataset_size / elapsed_time:.2f} samples/sec")