import torch
import numpy as np
import os
from smol.core import smol
from typing import List, Dict
import math

def parse_kwargs(argv):
    """Parses keyword arguments from sys.argv safely and converts numbers to appropriate types.

    Args:
        argv (list[str]): List of command-line arguments.

    Returns:
        dict: Parsed key-value pairs with numbers casted to int/float when possible.

    Raises:
        ValueError: If an argument is not in key=value format.
    """
    definition = {}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"Invalid argument format: '{arg}'. Expected format: key=value")
        
        key, value = arg.split("=", 1)
        if not key:
            raise ValueError(f"Missing key in argument: '{arg}'")

        # Try to cast to int, then float, otherwise keep as string
        if value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string if conversion fails

        definition[key] = value

    return definition

def inspect_gradients(model:torch.nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN value detected in {name} gradients in model {type(model).__name__}")
       
            grad_norm = param.grad.norm()

            if grad_norm < 1e-6:
                print(f"Vanishing gradient detected in {name} with norm {grad_norm} in model {type(model).__name__}")

            if grad_norm > 1e6:
                print(f"Exploding gradient detected in {name} with norm {grad_norm} in model {type(model).__name__}")

def calculate_psnr(max_value, mse):
    psnr = 10 * np.log10(max_value**2 / mse)
    return psnr

def save_tensor_to_path(root_dir:str, file_name:str, tensor:torch.Tensor) -> None:
    """
    Saves a tensor to a specified path after detaching it from the computation graph and moving it to the CPU.

    Args:
    - root_dir (str): The root directory where the tensor should be saved.
    - file_name (str): The name of the file to save the tensor as.
    - tensor (torch.Tensor): The tensor to be saved.

    Returns:
    - None
    """
    # Ensure the root directory exists
    os.makedirs(root_dir, exist_ok=True)
    
    # Construct the full path
    file_path = os.path.join(root_dir, f"{file_name}.pt")
    
    # Detach the tensor and move it to CPU
    tensor_to_save = tensor.detach().cpu()
    
    # Save the tensor
    torch.save(tensor_to_save, file_path)

def str_to_onehot(categories:list, 
                num_classes:int = smol.get_config("data", "CLASS_NUM"),
                class_enum:dict = smol.get_config("data", "CLASS_ENUM")):
    """
    Convert a list of category strings into a batch of one-hot encoded vectors.
    
    :param categories: List of strings representing categories.
    :return: A torch.Tensor of shape [len(categories), num_classes], 
             where each row is a one-hot vector.
    """
    one_hot_vecs = torch.zeros((len(categories), num_classes), dtype=torch.float32)
    
    for i, cat in enumerate(categories):
        class_index = class_enum[cat]
        one_hot_vecs[i, class_index - 1] = 1.0

    return one_hot_vecs.unsqueeze(2)

def onehot_to_str(one_hot_vecs: torch.Tensor, 
                  class_enum: Dict[str, int] = smol.get_config("data", "CLASS_ENUM")) -> List[str]:
    """
    Convert a batch of one-hot encoded vectors back to category strings.
    
    :param one_hot_vecs: A torch.Tensor of shape [-1, num_classes, 1].
    :param class_enum: Dictionary mapping category strings to class indices.
    :return: A list of strings representing the original categories.
    """
    if one_hot_vecs.dim() == 3 and one_hot_vecs.shape[2] == 1:
        one_hot_vecs = one_hot_vecs.squeeze(2)
    indices = torch.argmax(one_hot_vecs, dim=1) + 1
    index_to_class = {v: k for k, v in class_enum.items()}
    category_list = [index_to_class[idx.item()] for idx in indices]
    return category_list

import torch

def add_awgn_noise(x: torch.Tensor, target_snr_db: float, stats: np.ndarray) -> torch.Tensor:
    """
    Adds AWGN noise.
    
    Args:
        x (torch.Tensor): Original signal (B x N).
        target_snr_db (float): Desired SNR in decibels.
        stats (np.ndarray): Stats array where [:, 2] are mins and [:, 3] are maxs (same as in quantize).
    
    Returns:
        torch.Tensor: Noisy signal.
    """
    device = x.device
    mins = torch.tensor(stats[:, 2], dtype=x.dtype, device=device)
    maxs = torch.tensor(stats[:, 3], dtype=x.dtype, device=device)
    mins = mins.unsqueeze(0)
    maxs = maxs.unsqueeze(0)
    scale = maxs - mins
    scale[scale == 0] = 1e-8
    x_normalized = (x - mins) / scale
    signal_power = x_normalized.pow(2).mean()
    snr_linear = 10 ** (target_snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(x_normalized) * torch.sqrt(noise_power)
    x_noisy_normalized = x_normalized + noise
    x_noisy = x_noisy_normalized * scale + mins
    return x_noisy

def d1_psnr_pcc(ref_pc: torch.Tensor,
                test_pc: torch.Tensor,
                chunk: int | None = None,
                device="cpu") -> float:
    """
    JPEG-Pleno / MPEG-PCC-compliant D1 PSNR (point-to-point, symmetric).

    * ref_pc, test_pc : (1, 3, N) float or int tensors in voxel units
    * chunk           : chunk size for GPU memory saving
    * device          : 'cpu' or 'cuda'
    """
    ref  = ref_pc.to(device).squeeze(0).t().float()   # (N_ref, 3)
    test = test_pc.to(device).squeeze(0).t().float()  # (N_test, 3)
    ref  = torch.unique(ref , dim=0)
    test = torch.unique(test, dim=0)

    peak = ref.max()

    def coord_mse(a, b):
        # distance^2 / 3  (PCC definition)
        if chunk is None:
            d2 = torch.cdist(a, b).min(1).values.pow(2) / 3.0
            return d2.mean()
        mse, s = 0.0, 0
        while s < a.shape[0]:
            e   = s + chunk
            d2  = torch.cdist(a[s:e], b).min(1).values.pow(2) / 3.0
            mse += d2.sum()
            s   = e
        return mse / a.shape[0]

    mse_sym = torch.maximum(coord_mse(ref, test), coord_mse(test, ref)).item()
    return float('inf') if mse_sym == 0 else 10.0 * math.log10(peak**2 / mse_sym)