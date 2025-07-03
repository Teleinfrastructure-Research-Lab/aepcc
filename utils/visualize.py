import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from smol.core import smol
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from matplotlib import cm

def create_xy_grid(size=1.0, n=10):
    """
    Create a grid in the xy-plane.
    
    :param size: The total length of the grid.
    :param n: The number of divisions in the grid.
    :return: A list of LineSet geometries representing the grid.
    """
    lines = []
    for i in range(n + 1):
        x = -size / 2 + i * size / n
        lines.append([[x, -size / 2, 0], [x, size / 2, 0]])
        lines.append([[-size / 2, x, 0], [size / 2, x, 0]])

    grid_lines = []
    for line in lines:
        grid_lines.append(o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.array(line)),
            lines=o3d.utility.Vector2iVector([[0, 1]])
        ))

    return grid_lines

def visualize_point_cloud(point_cloud):
    """
    Visualizes a 3xN point cloud using matplotlib.

    Parameters:
    point_cloud (torch.Tensor): A 3xN PyTorch tensor representing the point cloud.
    """
    # Convert the tensor to numpy for matplotlib
    points = point_cloud.numpy()

    # Separate the x, y, z coordinates
    x = points[0, :]
    y = points[1, :]
    z = points[2, :]

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='viridis')

    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def visualize_point_cloud_o3d(point_cloud):
    """
    Visualizes a Nx3 or 3xN point cloud using Open3D.

    Parameters:
    point_cloud (torch.Tensor): A Nx3 or 3xN PyTorch tensor representing the point cloud.
    """
    # Ensure the tensor is on CPU
    if point_cloud.is_cuda:
        point_cloud = point_cloud.cpu()
    
    # Convert the tensor to numpy
    points = point_cloud.numpy()
    
    # Determine the shape and adjust if necessary
    if points.shape[0] == 3 and points.shape[1] >= 3:
        # Shape is (3, N), transpose to (N, 3)
        points = points.T
    elif points.shape[1] == 3 and points.shape[0] >= 3:
        # Shape is (N, 3), no need to transpose
        pass
    else:
        raise ValueError(f"Point cloud tensor must have shape (3, N) or (N, 3), but got {points.shape}")
    
    # Ensure the data type is float64 for Vector3dVector
    if points.dtype != np.float64:
        points = points.astype(np.float64)
    
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    try:
        pcd.points = o3d.utility.Vector3dVector(points)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to assign points to Open3D PointCloud: {e}")
    
    # Optional: Color the points based on the Z-axis using a colormap
    z = points[:, 2]
    z_min, z_max = z.min(), z.max()
    if z_max - z_min > 0:
        z_normalized = (z - z_min) / (z_max - z_min)  # Normalize to [0, 1]
    else:
        z_normalized = np.zeros_like(z)
    
    colormap = cm.get_cmap('viridis')
    colors = colormap(z_normalized)[:, :3]  # Ignore the alpha channel
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries(
        [pcd],
        window_name='Open3D Point Cloud',
        width=800,
        height=600,
        left=50,
        top=50,
        point_show_normal=False
    )

def visualize_two_point_clouds(pc1, pc2):
    """
    Visualizes two point clouds using matplotlib.

    Parameters:
    pc1 (torch.Tensor): A 3xN PyTorch tensor representing the first point cloud.
    pc2 (torch.Tensor): A 3xM PyTorch tensor representing the second point cloud.
    """
    # Convert the tensors to numpy for matplotlib
    points1 = pc1.numpy()
    points2 = pc2.numpy()

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the first point cloud
    ax.scatter(points1[0, :], points1[1, :], points1[2, :], color='blue', label='Point Cloud 1', alpha=0.6)

    # Plot the second point cloud
    ax.scatter(points2[0, :], points2[1, :], points2[2, :], color='green', label='Point Cloud 2', alpha=0.6)

    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    ax.legend()

    plt.show()

def visualize_multiple_point_clouds(point_clouds, additional_point_cloud):
    """
    Visualizes multiple 3xN point clouds and an additional 3xM point cloud using matplotlib.

    Parameters:
    point_clouds (torch.Tensor): A 3x3xN PyTorch tensor representing three point clouds.
    additional_point_cloud (torch.Tensor): A 3xM PyTorch tensor representing the additional point cloud.
    """
    # Convert the tensors to numpy for matplotlib
    points = point_clouds.numpy()
    additional_points = additional_point_cloud.numpy()

    # Create a 3D scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each of the 3 point clouds with different colors
    colors = ['r', 'g', 'b']
    for i in range(3):
        x = points[i, 0, :]
        y = points[i, 1, :]
        z = points[i, 2, :]
        ax.scatter(x, y, z, color=colors[i], label=f'Point Cloud {i+1}', alpha=0.6)

    # Plot the additional point cloud with a different color
    x_add = additional_points[0, :]
    y_add = additional_points[1, :]
    z_add = additional_points[2, :]
    ax.scatter(x_add, y_add, z_add, color='purple', label='Additional Point Cloud', alpha=0.8)

    # Set plot labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add a legend
    ax.legend()

    plt.show()