import torch
import os

from dataloaders.DataLoading import get_dataloader
from smol.core import smol

def test_single_dataloader(h5_path, batch_size=4, num_workers=0, split='train', max_points=1024, num_iterations=3):
    print("Testing regular dataloader...")

    dataloader = get_dataloader(h5_path, batch_size, num_workers, split, max_points)

    for iteration, batch in enumerate(dataloader, start=1):
        print(f"Iteration {iteration}/{num_iterations}:")
        print("Keys in the batch:", batch.keys())
        print("Point_clouds shape:", batch['point_clouds'].shape)  # [B, 3, N]
        print("Categories length:", len(batch['categories']))
        print("Masks shape:", batch['masks'].shape)  # [B, N]
        print("Paths length:", len(batch['paths']))

        # Verify tensor shapes and types
        assert batch['point_clouds'].shape[0] == batch_size, "Batch size doesn't match expected."
        assert batch['masks'].shape[0] == batch_size, "Masks batch size doesn't match expected."
        assert batch['point_clouds'].shape[1] == 3, "Point clouds should have 3 coordinates."
        assert batch['point_clouds'].dtype == torch.float32, "Point clouds should be float32."

        if iteration == num_iterations:
            break

    print("Regular dataloader test passed!\n")


if __name__ == "__main__":
    h5_path = smol.get_config("data", "H5_PATH_SYNTH")

    # Run tests for a few iterations
    test_single_dataloader(h5_path, batch_size=4, num_workers=0, split='train', max_points = smol.get_config('data', 'NUM_POINTS'), num_iterations=3)
