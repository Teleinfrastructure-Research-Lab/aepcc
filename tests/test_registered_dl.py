import torch
from tqdm import tqdm

from smol.core import smol
from smol.register.DataLoader import CommonDataLoader


def verify_tensor_shapes(tensor_dict, shape_dict):

    for key, expected_shape in shape_dict.items():
        if key not in tensor_dict:
            raise KeyError(f"Key '{key}' missing in tensor dictionary.")
        
        if type(tensor_dict[key]) == torch.Tensor:
            tensor_shape = tensor_dict[key].shape
            if len(tensor_shape) != len(expected_shape):
                raise ValueError(f"Shape mismatch for key '{key}': expected {expected_shape}, got {tensor_shape}")
            
            for actual_dim, expected_dim in zip(tensor_shape, expected_shape):
                if expected_dim != -1 and actual_dim != expected_dim:
                    raise KeyError(f"Mismatch in key '{key}': expected {expected_shape}, got {tensor_shape}")
            
def check_tensor_validity(tensor_dict):
    for key, tensor in tensor_dict.items():
        if type(tensor) == torch.Tensor:
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid values found in '{key}': contains NaN or Inf.")

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    smol.register(register_arch=False, register_loss=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    registry = smol.get_dl_registry()

    for name in tqdm(registry, total = len(registry)):
        try:
            output_def = registry[name].output_def
            dl = CommonDataLoader(name)
            _output = next(dl)
            verify_tensor_shapes(_output, output_def)
            check_tensor_validity(_output)
        except Exception as e:
            raise Exception(f"Test on dl {name} failed: {e}") from e
        