import pandas as pd
import os
import sys
import h5py
import open3d as o3d

from smol.core import smol
from utils.DataPipeline.h5 import write_split


if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    if len(sys.argv) <2:
        synth_flag = False
    else:
        synth_flag = sys.argv[1].lower() == "true"
    if synth_flag == False:
        metadata_csv_path = smol.get_config("data", "METADATA_FULL_CSV_PATH")
        h5_path = smol.get_config("data", "H5_PATH_FULL")
    else:
        metadata_csv_path = smol.get_config("data", "METADATA_SYNTH_CSV_PATH")
        h5_path = smol.get_config("data", "H5_PATH_SYNTH")
    num_points = smol.get_config("data", "SAMPLED_POINTS")
    aug_std = smol.get_config("data", "AUGMENTATION_STD")
    aug_radius = smol.get_config("data", "AUGMENTATION_RADIUS")

    metadata_df = pd.read_csv(metadata_csv_path, sep=",")
    metadata_train = metadata_df[metadata_df["split"] == "train"].reset_index(drop=True)
    num_train_samples = len(metadata_train)
    metadata_val = metadata_df[metadata_df["split"] == "val"].reset_index(drop=True)
    num_val_samples = len(metadata_val)
    metadata_test = metadata_df[metadata_df["split"] == "test"].reset_index(drop=True)
    num_test_samples = len(metadata_test)

    with h5py.File(h5_path, 'w') as h5f:
        string_dt = h5py.special_dtype(vlen=str)

        # TRAIN
        dataset_train_input = h5f.create_dataset(
            'train_input',
            shape=(num_train_samples, 4, num_points),
            chunks=(smol.get_config("data", "H5_SAMPLE_DIM_CHUNK_SIZE"), 4, num_points),
            compression=smol.get_config("data", "H5_COMPRESSION"),
            compression_opts=(
                smol.get_config("data", "H5_COMPRESSION_LEVEL") if smol.get_config("data", "H5_COMPRESSION") != "lzf" else None
            )
        )
        dataset_train_label = h5f.create_dataset(
            'train_label',
            shape=(num_train_samples,),
            dtype=string_dt,
            chunks=(smol.get_config("data", "H5_SAMPLE_DIM_CHUNK_SIZE"),),
            compression=smol.get_config("data", "H5_COMPRESSION"),
            compression_opts=(
                smol.get_config("data", "H5_COMPRESSION_LEVEL") if smol.get_config("data", "H5_COMPRESSION") != "lzf" else None
            )
        )
        write_split(dataset_train_input, dataset_train_label, metadata_train, num_points, aug_std, aug_radius)

        # VALIDATION
        dataset_val = h5f.create_dataset(
            'val',
            shape=(num_val_samples, 4, num_points),
            chunks=(smol.get_config("data", "H5_SAMPLE_DIM_CHUNK_SIZE"), 4, num_points),
            compression=smol.get_config("data", "H5_COMPRESSION"),
            compression_opts=(
                smol.get_config("data", "H5_COMPRESSION_LEVEL") if smol.get_config("data", "H5_COMPRESSION") != "lzf" else None
            )
        )
        dataset_val_label = h5f.create_dataset(
            'val_label',
            shape=(num_val_samples,),
            dtype=string_dt,
            chunks=(smol.get_config("data", "H5_SAMPLE_DIM_CHUNK_SIZE"),),
            compression=smol.get_config("data", "H5_COMPRESSION"),
            compression_opts=(
                smol.get_config("data", "H5_COMPRESSION_LEVEL") if smol.get_config("data", "H5_COMPRESSION") != "lzf" else None
            )
        )
        write_split(dataset_val, dataset_val_label, metadata_val, num_points, aug_std, aug_radius)

        # TEST
        dataset_test = h5f.create_dataset(
            'test',
            shape=(num_test_samples, 4, num_points),
            chunks=(smol.get_config("data", "H5_SAMPLE_DIM_CHUNK_SIZE"), 4, num_points),
            compression=smol.get_config("data", "H5_COMPRESSION"),
            compression_opts=(
                smol.get_config("data", "H5_COMPRESSION_LEVEL") if smol.get_config("data", "H5_COMPRESSION") != "lzf" else None
            )
        )
        dataset_test_label = h5f.create_dataset(
            'test_label',
            shape=(num_test_samples,),
            dtype=string_dt,
            chunks=(smol.get_config("data", "H5_SAMPLE_DIM_CHUNK_SIZE"),),
            compression=smol.get_config("data", "H5_COMPRESSION"),
            compression_opts=(
                smol.get_config("data", "H5_COMPRESSION_LEVEL") if smol.get_config("data", "H5_COMPRESSION") != "lzf" else None
            )
        )
        write_split(dataset_test, dataset_test_label, metadata_test, num_points, aug_std, aug_radius)
    