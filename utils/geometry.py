import numpy as np
import torch
from smol.core import smol
import math

def apply_random_rotation_numpy(point_cloud: np.ndarray) -> np.ndarray:
    """
    Applies a random rotation to the point cloud.

    Args:
        point_cloud (np.ndarray): An Nx3 array representing the point cloud.

    Returns:
        np.ndarray: The rotated point cloud.
    """
    angles = np.random.rand(3) * 2 * np.pi  # Random angles between 0 and 2π

    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ])
    rot_y = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])
    rot_z = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])
    
    rotation_matrix = rot_z @ rot_y @ rot_x
    rotated_point_cloud = point_cloud @ rotation_matrix.T
    return rotated_point_cloud

def remove_points_in_radius_numpy(point_cloud: np.ndarray, radius: float) -> np.ndarray:
    """
    Removes all points within a radius around a randomly selected point.

    Args:
        point_cloud (np.ndarray): An Nx3 array representing the point cloud.
        radius (float): The radius around the selected point.

    Returns:
        np.ndarray: The point cloud with points within the radius removed.
    """
    random_point = point_cloud[np.random.randint(0, point_cloud.shape[0])]
    distances = np.linalg.norm(point_cloud - random_point, axis=1)
    return point_cloud[distances > radius]

def remove_points_fuzzy_sphere_numpy(point_cloud: np.ndarray, radius: float, sigma: float = 0.1) -> np.ndarray:
    """
    Removes points within a spherical region around a randomly selected point with a fuzzy boundary.
    """
    if point_cloud.shape[0] == 0:
        print("Warning: Empty point cloud detected.")
        return point_cloud

    random_idx = np.random.randint(0, point_cloud.shape[0])
    random_point = point_cloud[random_idx]
    distances = np.linalg.norm(point_cloud - random_point, axis=1)
    probabilities = 1 / (1 + np.exp(-(distances - radius) / sigma))
    random_probs = np.random.rand(len(probabilities))
    return point_cloud[random_probs > probabilities]

def compute_scale_numpy(points: np.ndarray) -> float:
    """
    Computes the scale of the point cloud as the maximum distance from the centroid.
    """
    if points.size == 0:
        return 1.0
    centroid = np.mean(points, axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    scale = np.max(distances)
    return scale if scale > 0 else 1.0

def add_random_noise_numpy(point_cloud: np.ndarray, std: float = 0.01) -> np.ndarray:
    """
    Adds random Gaussian noise proportional to the scale of the point cloud.
    """
    scale = compute_scale_numpy(point_cloud)
    noise_std = scale * std
    noise = np.random.randn(*point_cloud.shape) * noise_std
    return point_cloud + noise

def sphere_normalization_numpy(points: np.ndarray) -> np.ndarray:
    """
    Normalizes a point cloud to fit inside a unit sphere.
    """
    if points.size == 0:
        print("Warning: Empty point cloud detected.")
        return points
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    distances = np.linalg.norm(centered_points, axis=1)
    max_distance = np.max(distances)
    
    if max_distance == 0:
        print("Warning: Degenerate point cloud with zero max distance detected.")
        return points
    return centered_points / max_distance

def apply_random_rotation(point_cloud:torch.Tensor):
    """
    Applies a random rotation to the point cloud.

    Args:
        point_cloud (torch.Tensor): An Nx3 tensor representing the point cloud.

    Returns:
        torch.Tensor: The rotated point cloud.
    """
    # Generate random rotation angles (in radians) for each axis
    angles = torch.rand(3) * 2 * torch.pi  # Random angles between 0 and 2π

    # Rotation matrices
    rot_x = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(angles[0]), -torch.sin(angles[0])],
        [0, torch.sin(angles[0]), torch.cos(angles[0])]
    ])
    rot_y = torch.tensor([
        [torch.cos(angles[1]), 0, torch.sin(angles[1])],
        [0, 1, 0],
        [-torch.sin(angles[1]), 0, torch.cos(angles[1])]
    ])
    rot_z = torch.tensor([
        [torch.cos(angles[2]), -torch.sin(angles[2]), 0],
        [torch.sin(angles[2]), torch.cos(angles[2]), 0],
        [0, 0, 1]
    ])

    # Combine rotations into a single rotation matrix
    rotation_matrix = rot_z @ rot_y @ rot_x

    # Apply the rotation to the point cloud
    rotated_point_cloud = point_cloud @ rotation_matrix.T
    return rotated_point_cloud

def remove_points_in_radius(point_cloud:torch.Tensor, radius:float):
    """
    Removes all points within a radius around a randomly selected point.

    Args:
        point_cloud (torch.Tensor): An Nx3 tensor representing the point cloud.
        radius (float): The radius around the selected point.

    Returns:
        torch.Tensor: The point cloud with points within the radius removed.
    """
    # Select a random point from the point cloud
    random_point = point_cloud[torch.randint(0, point_cloud.size(0), (1,))]

    # Compute distances of all points from the selected point
    distances = torch.norm(point_cloud - random_point, dim=1)

    # Keep points outside the radius
    filtered_point_cloud = point_cloud[distances > radius]
    return filtered_point_cloud

def remove_points_fuzzy_sphere(point_cloud: torch.Tensor, radius: float, sigma: float = 0.1) -> torch.Tensor:
    """
    Removes points within a spherical region around a randomly selected point with a fuzzy boundary.
    Points closer to the center are more likely to be removed, and points near the boundary have a probability of being removed.

    Args:
        point_cloud (torch.Tensor): An Nx3 tensor representing the point cloud.
        radius (float): The radius of the sphere defining the removal region.
        sigma (float): Controls the smoothness of the transition (fuzziness).
                       Smaller values make the transition sharper.

    Returns:
        torch.Tensor: The point cloud with points probabilistically removed.
    """
    if point_cloud.size(0) == 0:
        print("Warning: Empty point cloud detected.")
        return point_cloud

    # Select a random point from the point cloud
    random_idx = torch.randint(0, point_cloud.size(0), (1,))
    random_point = point_cloud[random_idx]  # Shape: (1, 3)

    # Compute distances of all points from the selected point
    distances = torch.norm(point_cloud - random_point, dim=1)  # Shape: (N,)

    # Compute removal probabilities using a sigmoid function
    # Points with distance < radius have higher probability of removal
    probabilities = torch.sigmoid(-(distances - radius) / sigma)  # Shape: (N,)

    # Generate random numbers for each point
    random_probs = torch.rand_like(probabilities)

    # Keep points where random_probs > probabilities (i.e., do not remove)
    mask = random_probs > probabilities  # Shape: (N,)
    filtered_point_cloud = point_cloud[mask]
    
    return filtered_point_cloud

def compute_scale(points: torch.Tensor) -> float:
    """
    Computes the scale of the point cloud as the maximum distance from the centroid.
    
    Args:
        points (torch.Tensor): An Nx3 tensor representing the point cloud.
    
    Returns:
        float: The scale of the point cloud.
    """
    if points.numel() == 0:
        return 1.0  # Avoid division by zero; adjust as needed
    centroid = torch.mean(points, dim=0, keepdim=True)
    centered_points = points - centroid
    distances = torch.norm(centered_points, dim=1)
    scale = torch.max(distances).item()
    return scale if scale > 0 else 1.0  # Avoid zero scale

def add_random_noise(point_cloud:torch.Tensor, std:float=0.01):
    """
    Adds random Gaussian noise proportional to the scale of the point cloud.
    
    Args:
        point_cloud (torch.Tensor): An Nx3 tensor representing the point cloud.
        scale_factor (float): The fraction of the scale to use as the std of the noise.
    
    Returns:
        torch.Tensor: The point cloud with added noise.
    """
    scale = compute_scale(point_cloud)
    noise_std = scale * std
    noise = torch.randn_like(point_cloud) * noise_std
    noisy_point_cloud = point_cloud + noise
    return noisy_point_cloud

def sphere_normalization(points):
    """
    Normalizes a point cloud to fit inside a unit sphere.
    
    Args:
        points (torch.Tensor): A tensor of shape (N, 3) where N is the number of points.

    Returns:
        torch.Tensor: The normalized points of shape (N, 3).
                     If the input tensor is empty, returns the input as is.
    """
    if points.numel() == 0:
        print("Warning: Empty point cloud detected.")
        return points
    centroid = torch.mean(points, dim=0, keepdim=True)  # shape: (1, 3)
    centered_points = points - centroid  # shape: (N, 3)
    distances = torch.sqrt(torch.sum(centered_points ** 2, dim=1))  # shape: (N,)
    max_distance = torch.max(distances)  # scalar
    
    if max_distance == 0:
        print("Warning: Degenerate point cloud with zero max distance detected.")
        return points
    normalized_points = centered_points / max_distance  # shape: (N, 3)
    
    return normalized_points

def normalize_points(points):
    # Calculate the minimum and maximum values along each axis
    min_vals = np.min(points, axis=0)
    max_vals = np.max(points, axis=0)
    
    # Find the maximum range (max - min) across all axes to maintain aspect ratio
    max_range = np.max(max_vals - min_vals)
    
    # Scale points to have a maximum range of 2 (from -1 to 1) uniformly across all axes
    # First, center the points around 0 by subtracting the midpoint
    midpoint = (max_vals + min_vals) / 2.0
    points_centered = points - midpoint
    
    # Then, scale based on the maximum range
    # We divide by max_range / 2 to scale to [-1, 1] because max_range corresponds to a range of 2 in the target space
    norm_points = points_centered / (max_range / 2.0)
    
    return norm_points

def sphere_normalization_with_params(points):
    """
    Normalizes a point cloud to fit inside a unit sphere and returns normalization parameters.
    
    Args:
        points (torch.Tensor): A tensor of shape (N, 3) where N is the number of points.

    Returns:
        tuple: (normalized_points, centroid, max_distance)
               - normalized_points: The normalized points of shape (N, 3).
               - centroid: The centroid of the original point cloud (1, 3).
               - max_distance: The maximum distance of points from the centroid (scalar).
    """
    if points.numel() == 0:
        print("Warning: Empty point cloud detected.")
        return points, None, None
    
    centroid = torch.mean(points, dim=0, keepdim=True)  # shape: (1, 3)
    centered_points = points - centroid  # shape: (N, 3)
    distances = torch.sqrt(torch.sum(centered_points ** 2, dim=1))  # shape: (N,)
    max_distance = torch.max(distances)  # scalar
    
    if max_distance == 0:
        print("Warning: Degenerate point cloud with zero max distance detected.")
        return points, centroid, max_distance
    
    normalized_points = centered_points / max_distance  # shape: (N, 3)
    
    return normalized_points, centroid, max_distance

def sphere_normalization_masked_with_params(points: torch.Tensor, mask: torch.Tensor):

    """
    Same like sphere_normalization_with_params, but works only on masked points (mask==1).
    """

    if points.numel() == 0:
        print("Warning: Empty point cloud detected.")
        return points, None, None

    if mask.dim() == 2:
        mask = mask.squeeze(0)
    mask_bool = mask.bool()

    if mask_bool.sum() == 0:
        print("Warning: Mask contains no valid points — nothing to normalize.")
        return points, None, None
    valid_pts = points[mask_bool]
    centroid = valid_pts.mean(dim=0, keepdim=True)
    centered = valid_pts - centroid
    distances = torch.linalg.norm(centered, dim=1)
    max_distance = distances.max()

    if max_distance == 0:
        print("Warning: Degenerate valid set with zero max distance detected.")
        return points, centroid, max_distance
    normalized_valid = centered / max_distance
    normalized_points = points.clone()
    normalized_points[mask_bool] = normalized_valid

    return normalized_points, centroid, max_distance

def sphere_denormalization(normalized_points, centroid, max_distance):
    """
    Denormalizes a point cloud using stored centroid and max_distance values.
    
    Args:
        normalized_points (torch.Tensor): A tensor of shape (N, 3) containing normalized points.
        centroid (torch.Tensor): The centroid of the original point cloud (1, 3).
        max_distance (torch.Tensor or float): The maximum distance of points from the centroid.
    
    Returns:
        torch.Tensor: The denormalized points of shape (N, 3).
    """
    if centroid is None or max_distance is None:
        raise ValueError("Invalid centroid or max_distance values for denormalization.")
    
    denormalized_points = (normalized_points * max_distance) + centroid  # shape: (N, 3)
    return denormalized_points

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    #PYTORCH3D implementation
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    #PYTORCH3D implementation
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)

def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    #PYTORCH3D implementation
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)

def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    #PYTORCH3D implementation
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling

def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    #PYTORCH3D implementation
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]

def generate_sphere_encoding(num_points:int, scale:float = 1.0) -> torch.Tensor:
    """ Generates uniform points on a sphere using Fibonacci lattice. """
    indices = torch.linspace(0.5, num_points - 0.5, num_points)
    phi = math.pi * (3 - math.sqrt(5))
    theta = torch.acos(1 - 2 * indices / num_points)
    longitude = phi * indices

    x = torch.cos(longitude) * torch.sin(theta)
    y = torch.sin(longitude) * torch.sin(theta)
    z = torch.cos(theta)

    return torch.stack([x, y, z])*scale  # (3, num_points)

def generate_gaussian_encoding(num_points:int, mean:float=0.0, std:float=1.0) -> torch.Tensor:
    """ Generates random Gaussian-distributed points in 3D space. """
    return torch.randn((3, num_points)) * std + mean  # (3, num_points)

def generate_grid_encoding(grid_size:int, max_value:float):
    """ Generates a uniform 2D grid. """
    x = torch.linspace(-max_value, max_value, grid_size)
    y = torch.linspace(-max_value, max_value, grid_size)
    mesh_x, mesh_y = torch.meshgrid(x, y, indexing="ij")

    return torch.stack([mesh_x.flatten(), mesh_y.flatten()]) 

def random_subsample_points(tensor: torch.Tensor, mask: torch.Tensor, num_points: int):
    """
    Randomly subsamples points along dimension 2 of a B x 3 x N tensor,
    prioritizing points with mask value 1.

    Args:
        tensor (torch.Tensor): Input tensor of shape (B, 3, N).
        mask (torch.Tensor): Mask tensor of shape (B, N), indicating valid points.
        num_points (int): Number of points to subsample.

    Returns:
        torch.Tensor: Indices of subsampled points.
        torch.Tensor: Subsampled tensor of shape (B, 3, num_points).
    """
    B, C, N = tensor.shape
    if num_points > N:
        raise ValueError(f"num_points {num_points} exceeds available points {N}")

    subsampled_indices = []

    for b in range(B):
        # Extract valid and invalid indices for the current batch
        valid_indices = torch.nonzero(mask[b] == 1, as_tuple=False).squeeze()
        invalid_indices = torch.nonzero(mask[b] == 0, as_tuple=False).squeeze()

        # Ensure valid_indices is a 1D tensor
        if valid_indices.dim() == 0:
            valid_indices = valid_indices.unsqueeze(0)

        # Sample points
        sampled_valid = valid_indices[torch.randperm(len(valid_indices))[:num_points]]
        
        if len(sampled_valid) < num_points:
            remaining_points = num_points - len(sampled_valid)
            sampled_invalid = invalid_indices[torch.randperm(len(invalid_indices))[:remaining_points]]
            sampled_indices = torch.cat((sampled_valid, sampled_invalid))
        else:
            sampled_indices = sampled_valid
        
        subsampled_indices.append(sampled_indices)

    subsampled_indices = torch.stack(subsampled_indices)
    subsampled_tensor = torch.gather(
        tensor, 2, subsampled_indices.unsqueeze(1).expand(-1, C, -1)
    )

    return subsampled_indices, subsampled_tensor