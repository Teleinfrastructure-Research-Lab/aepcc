import torch
import zlib
import io
import cbor2
import numpy as np

from smol.signal import Signal

def max_distance_masked(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the maximum Euclidean distance between points marked by the mask.

    Args:
        points (torch.Tensor): A tensor of shape (3, N) representing N points in 3D space.
        mask (torch.Tensor): A boolean or integer tensor of shape (N,), where 1 indicates the points to consider.

    Returns:
        float: The maximum Euclidean distance between selected points.
    """
    selected_points = points[:, mask == 1]

    if selected_points.shape[1] < 2:
        return torch.tensor(0.0)

    pairwise_diff = selected_points[:, :, None] - selected_points[:, None, :]
    pairwise_distances = torch.norm(pairwise_diff, dim=0)

    max_distance = pairwise_distances.max()

    return max_distance.item()

def max_abs_coordinate_masked(points: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Computes the maximum absolute coordinate value (in any dimension) among the masked points.

    Args:
        points (torch.Tensor): A tensor of shape (3, N) representing N points in 3D space.
        mask (torch.Tensor): A boolean or integer tensor of shape (N,), where 1 indicates the points to consider.

    Returns:
        float: The maximum absolute coordinate value among the selected points.
    """
    selected_points = points[:, mask == 1]

    if selected_points.numel() == 0:
        return 0.0

    max_abs = selected_points.abs().max()

    return max_abs.item()

def bounding_box_diagonal_masked(points: torch.Tensor, mask: torch.Tensor) -> float:
    """
    Computes the diagonal length of the bounding box enclosing the masked points.

    Args:
        points (torch.Tensor): A tensor of shape (3, N) representing N points in 3D space.
        mask (torch.Tensor): A boolean or integer tensor of shape (N,), where 1 indicates the points to consider.

    Returns:
        float: The length of the diagonal of the bounding box around the selected points.
    """
    selected_points = points[:, mask == 1]

    if selected_points.numel() == 0:
        return 0.0

    min_coords = selected_points.min(dim=1).values
    max_coords = selected_points.max(dim=1).values

    diagonal = torch.norm(max_coords - min_coords, p=2)

    return diagonal.item()
    

def remove_key(original_dict, key_to_remove):
    """
    Returns a new dictionary with all keys from original_dict except key_to_remove.

    Args:
        original_dict (dict): The original dictionary.
        key_to_remove (any): The key to exclude from the new dictionary.

    Returns:
        dict: A new dictionary without key_to_remove.
    """
    return {k: v for k, v in original_dict.items() if k != key_to_remove}

def remove_key_from_signal(original_signal: Signal, key_to_remove:str):
    """
    Returns a new dictionary with all keys from original_dict except key_to_remove.

    Args:
        original_signal (Signal): The original dictionary.
        key_to_remove (any): The key to exclude from the new dictionary.

    Returns:
        signal: A new signal without key_to_remove.
    """
    signal_items = original_signal.items()
    signal_def = original_signal.signal_def.items()
    signal_items = {k: v for k, v in signal_items if k != key_to_remove}
    signal_def = {k: v for k, v in signal_def if k != key_to_remove}

    return Signal(signal_items, signal_def)

def compress_object(data: dict) -> bytes:
    """
    Compresses a dictionary of PyTorch tensors using CBOR and zlib.

    Args:
        data (dict): A dictionary where values are PyTorch tensors.

    Returns:
        bytes: The compressed binary data.
    """
    serializable_data = {}

    for key, tensor in data.items():
        if isinstance(tensor, torch.Tensor):
            numpy_array = tensor.squeeze(0).cpu().numpy()
            serializable_data[key] = {
                "shape": numpy_array.shape,
                "dtype": str(numpy_array.dtype),
                "data": numpy_array.tobytes()
            }
        else:
            raise ValueError("All values in the dictionary must be PyTorch tensors.")

    return zlib.compress(cbor2.dumps(serializable_data))

def compress_scene(scene_imr: list) -> bytes:
    """
    Compresses a list of dictionaries containing PyTorch tensors using CBOR and zlib.

    Args:
        scene_imr (list): A list of dictionaries where values are PyTorch tensors.

    Returns:
        bytes: The compressed binary data.
    """
    serializable_scene_objs = []
    
    for scene_obj in scene_imr:
        serializable_data = {}
        
        for key, tensor in scene_obj.items():
            if isinstance(tensor, torch.Tensor):
                numpy_array = tensor.squeeze(0).detach().cpu().numpy()
                serializable_data[key] = {
                    "shape": numpy_array.shape,
                    "dtype": str(numpy_array.dtype),
                    "data": numpy_array.tobytes()
                }
            else:
                raise ValueError("All values in each dictionary must be PyTorch tensors.")
        
        serializable_scene_objs.append(serializable_data)
    
    return zlib.compress(cbor2.dumps(serializable_scene_objs))


def decompress_object(compressed_data: bytes) -> dict:
    """
    Decompresses a zlib-compressed CBOR dictionary back into PyTorch tensors.

    Args:
        compressed_data (bytes): The compressed binary data.

    Returns:
        dict: The decompressed dictionary where values are PyTorch tensors.
    """
    decompressed_data = {}
    cbor_data = cbor2.loads(zlib.decompress(compressed_data))

    for key, tensor_info in cbor_data.items():
        shape = tuple(tensor_info["shape"])
        dtype = np.dtype(tensor_info["dtype"])
        numpy_array = np.frombuffer(tensor_info["data"], dtype=dtype).reshape(shape)
        decompressed_data[key] = torch.from_numpy(numpy_array.copy()).unsqueeze(0)

    return decompressed_data

def decompress_scene(compressed_data: bytes) -> list:
    """
    Decompresses a zlib-compressed CBOR list of dictionaries back into PyTorch tensors.

    Args:
        compressed_data (bytes): The compressed binary data.

    Returns:
        list: A list of decompressed dictionaries where values are PyTorch tensors.
    """
    decompressed_scene_objs = []
    cbor_data = cbor2.loads(zlib.decompress(compressed_data))
    
    for scene_obj in cbor_data:
        decompressed_data = {}
        
        for key, tensor_info in scene_obj.items():
            shape = tuple(tensor_info["shape"])
            dtype = np.dtype(tensor_info["dtype"])
            numpy_array = np.frombuffer(tensor_info["data"], dtype=dtype).reshape(shape)
            decompressed_data[key] = torch.from_numpy(numpy_array.copy()).unsqueeze(0)
        
        decompressed_scene_objs.append(decompressed_data)
    
    return decompressed_scene_objs

def quantize8_tensor(tensor: torch.Tensor, stats: np.ndarray) -> torch.Tensor:
    device = tensor.device
    tensor_np = tensor.detach().cpu().numpy()
    mins = stats[:, 2]
    maxs = stats[:, 3]
    scales = (maxs - mins) / 255.0
    scales[scales == 0] = 1e-8
    quantized = ((tensor_np - mins) / scales).round().clip(0, 255).astype(np.uint8)
    return torch.from_numpy(quantized).to(device)


def dequantize8_tensor(quantized_tensor: torch.Tensor, stats: np.ndarray) -> torch.Tensor:
    device = quantized_tensor.device
    q_np = quantized_tensor.detach().cpu().numpy().astype(np.float32)
    mins = stats[:, 2]
    maxs = stats[:, 3]
    scales = (maxs - mins) / 255.0
    scales[scales == 0] = 1e-8
    dequantized = q_np * scales + mins
    return torch.from_numpy(dequantized).to(device)

def quantize_and_pack_4bit(tensor: torch.Tensor, stats: np.ndarray) -> torch.Tensor:
    device = tensor.device
    tensor_np = tensor.detach().cpu().numpy()

    mins = stats[:, 2]
    maxs = stats[:, 3]

    qmax = 15
    scales = (maxs - mins) / qmax
    scales[scales == 0] = 1e-8
    quantized = ((tensor_np - mins) / scales).round().clip(0, qmax).astype(np.uint8)
    flat = quantized.flatten()
    if flat.size % 2 != 0:
        flat = np.append(flat, 0)
    packed = (flat[::2] << 4) | flat[1::2]
    return torch.from_numpy(packed.astype(np.uint8)).unsqueeze(0).to(device)

def unpack_and_dequantize_4bit(packed_tensor: torch.Tensor, stats: np.ndarray, original_shape: tuple) -> torch.Tensor:
    device = packed_tensor.device
    packed_np = packed_tensor.detach().cpu().numpy()
    high = (packed_np >> 4) & 0x0F
    low = packed_np & 0x0F
    quantized_flat = np.empty(high.size * 2, dtype=np.uint8)
    quantized_flat[::2] = high
    quantized_flat[1::2] = low
    total_elements = np.prod(original_shape)
    quantized_flat = quantized_flat[:total_elements]
    quantized = quantized_flat.reshape(original_shape)
    mins = stats[:, 2]
    maxs = stats[:, 3]
    qmax = 15
    scales = (maxs - mins) / qmax
    scales[scales == 0] = 1e-8

    dequantized = quantized.astype(np.float32) * scales + mins
    return torch.from_numpy(dequantized).to(device)

def quantize_and_pack_2bit(tensor: torch.Tensor, stats: np.ndarray) -> torch.Tensor:
    device = tensor.device
    tensor_np = tensor.detach().cpu().numpy()
    B, N = tensor_np.shape
    mins = stats[:, 2]
    maxs = stats[:, 3]
    qmax = 3
    scales = (maxs - mins) / qmax
    scales[scales == 0] = 1e-8
    quantized = ((tensor_np - mins) / scales).round().clip(0, qmax).astype(np.uint8)
    flat = quantized.flatten()
    orig_len = flat.size
    if orig_len % 4 != 0:
        pad_len = 4 - (orig_len % 4)
        flat = np.append(flat, [0] * pad_len)
    packed = (
        (flat[0::4] << 6) |
        (flat[1::4] << 4) |
        (flat[2::4] << 2) |
        flat[3::4]
    )

    return torch.from_numpy(packed.astype(np.uint8)).unsqueeze(0).to(device)

def unpack_and_dequantize_2bit(packed_tensor: torch.Tensor, stats: np.ndarray, original_shape: tuple) -> torch.Tensor:
    device = packed_tensor.device
    packed_np = packed_tensor.detach().cpu().numpy().flatten()
    v0 = (packed_np >> 6) & 0x03
    v1 = (packed_np >> 4) & 0x03
    v2 = (packed_np >> 2) & 0x03
    v3 = packed_np & 0x03
    quantized_flat = np.empty(len(packed_np) * 4, dtype=np.uint8)
    quantized_flat[0::4] = v0
    quantized_flat[1::4] = v1
    quantized_flat[2::4] = v2
    quantized_flat[3::4] = v3
    total_elements = original_shape[0] * original_shape[1]
    quantized_flat = quantized_flat[:total_elements]
    quantized = quantized_flat.reshape(original_shape)
    mins = stats[:, 2]
    maxs = stats[:, 3]
    qmax = 3
    scales = (maxs - mins) / qmax
    scales[scales == 0] = 1e-8
    dequantized = quantized.astype(np.float32) * scales + mins
    return torch.from_numpy(dequantized).to(device)

def quantize_and_pack_1bit(tensor: torch.Tensor, stats: np.ndarray) -> torch.Tensor:
    device = tensor.device
    tensor_np = tensor.detach().cpu().numpy()
    B, N = tensor_np.shape
    mins = stats[:, 2]
    maxs = stats[:, 3]
    midpoints = (mins + maxs) / 2
    quantized = (tensor_np > midpoints).astype(np.uint8)
    flat = quantized.flatten()
    orig_len = flat.size
    if orig_len % 8 != 0:
        pad_len = 8 - (orig_len % 8)
        flat = np.append(flat, [0] * pad_len)
    packed = np.packbits(flat)
    return torch.from_numpy(packed).unsqueeze(0).to(device)

def unpack_and_dequantize_1bit(packed_tensor: torch.Tensor, stats: np.ndarray, original_shape: tuple) -> torch.Tensor:
    device = packed_tensor.device
    packed_np = packed_tensor.detach().cpu().numpy().flatten()
    unpacked = np.unpackbits(packed_np)
    total_elements = original_shape[0] * original_shape[1]
    unpacked = unpacked[:total_elements]
    quantized = unpacked.reshape(original_shape).astype(np.uint8)
    mins = stats[:, 2]
    maxs = stats[:, 3]
    midpoints = (mins + maxs) / 2
    dequantized = np.where(quantized == 1, maxs, mins).astype(np.float32)
    return torch.from_numpy(dequantized).to(device)

def remove_isolated_points(points: torch.Tensor,
                           edges: torch.Tensor):
    """
    Remove isolated (degree-0) points from a point cloud,
    based on absence from edges[1] (i.e., never being a destination).

    Parameters
    ----------
    points : torch.Tensor
        Shape (B, 3, N) **or** (3, N).  
        Coordinates of N points; B is an optional batch dimension.
    edges : torch.LongTensor
        Shape (2, E).  
        Each column stores the (src_idx, dst_idx) of an edge.  Indices
        must refer to the last dimension of `points`.

    Returns
    -------
    filtered : torch.Tensor
        Same rank as `points`, but with the isolated columns removed.
    """
    dst_indices = edges[1].to(torch.long)  # Ensure indices are long/int64
    used = torch.unique(dst_indices)
    used, _ = torch.sort(used)
    N = points.shape[-1]
    mask = torch.zeros(N, dtype=torch.bool, device=points.device)
    mask[used] = True
    filtered = points[..., mask]
    return filtered
