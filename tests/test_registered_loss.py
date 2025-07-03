import torch
from tqdm import tqdm

from smol.core import smol
from smol.register.Loss import CommonLoss
from smol.signal import Signal


def generate_random_tensors(signal_def, replace_value=4):

    random_tensors = {}

    for key, shape in signal_def.items():
        new_shape = tuple(replace_value if dim == -1 else dim for dim in shape)
        if key == "mask" or key =="masks":
            random_tensors[key] = torch.randint(0, 2, new_shape)
        else:
            random_tensors[key] = torch.randn(new_shape, requires_grad=True)

    return random_tensors

def verify_tensor_shapes(tensor_dict, shape_dict):

    for key, expected_shape in shape_dict.items():
        if key not in tensor_dict:
            raise KeyError(f"Key '{key}' missing in tensor dictionary.")
        
        tensor_shape = tensor_dict[key].shape
        if len(tensor_shape) != len(expected_shape):
            raise ValueError(f"Shape mismatch for key '{key}': expected {expected_shape}, got {tensor_shape}")
        
        for actual_dim, expected_dim in zip(tensor_shape, expected_shape):
            if expected_dim != -1 and actual_dim != expected_dim:
                raise KeyError(f"Mismatch in key '{key}': expected {expected_shape}, got {tensor_shape}")
            
def check_tensor_validity(tensor_dict):
    for key, tensor in tensor_dict.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Invalid values found in '{key}': contains NaN or Inf.")

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    smol.register(register_arch=False, register_dl=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    registry = smol.get_loss_registry()

    for name in tqdm(registry, total = len(registry)):
        try:
            input_def = registry[name].input_def
            output_def = registry[name].output_def
            _input = Signal(generate_random_tensors(input_def), input_def)
            _input = _input.to(device)
            loss = CommonLoss(name)
            _output = loss(_input)
            verify_tensor_shapes(_output, output_def)
            check_tensor_validity(_output)
            _output.backward()
        except Exception as e:
            raise Exception(f"Test on loss {name} failed: {e}") from e
    