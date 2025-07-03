import zipfile
import io
import numpy as np
import open3d as o3d
import torch

from smol.core import smol
from architecture.folding import FoldingEncoder, FoldingDecoder
from architecture.gae import GraphEncoder, GraphDecoder
from architecture.tgae import TransformerPCEncoder, TransformerPCDecoder
from utils.geometry import sphere_normalization_numpy

def filter_by_epoch(namelist, epoch):
    suffix = f"_{epoch}.pth"
    return [s for s in namelist if s.endswith(suffix)]

def extract_submodule_state_dict(state_dict, submodule_name):
    prefix = f"{submodule_name}."
    sub_dict = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    return sub_dict

def load_checkpoint_from_zip(zip_path: str, epoch: str):
    """ Load a single checkpoint from the ZIP archive """
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        all_checkpoints = zipf.namelist()
        file_name = filter_by_epoch(all_checkpoints, epoch)[0]
        with zipf.open(file_name) as f:
            buffer = io.BytesIO(f.read())
            state_dict = torch.load(buffer, map_location='cpu')
            return state_dict

def get_fne(vector_size):
    fne = FoldingEncoder(vector_size)
    state_dict = state_dict = torch.load(f"data/models/folding-net{vector_size}.pth", map_location='cpu')
    encoder_dict = extract_submodule_state_dict(state_dict, "encoder")
    fne.load_state_dict(encoder_dict)
    return fne

def get_ge(vector_size):
    ge = GraphEncoder(vector_size, film_flag = False)
    state_dict = state_dict = torch.load(f"data/models/gae{vector_size}.pth", map_location='cpu')
    encoder_dict = extract_submodule_state_dict(state_dict, "encoder")
    ge.load_state_dict(encoder_dict)
    return ge

def get_tge(vector_size):
    tge = TransformerPCEncoder(vector_size, num_layers=1, film_flag = False)
    state_dict = state_dict = torch.load(f"data/models/tgae{vector_size}.pth", map_location='cpu')
    encoder_dict = extract_submodule_state_dict(state_dict, "encoder")
    tge.load_state_dict(encoder_dict)
    return tge

def get_fnd(vector_size):
    fnd = FoldingDecoder(vector_size, "sphere")
    state_dict = state_dict = torch.load(f"data/models/folding-net{vector_size}.pth", map_location='cpu')
    encoder_dict = extract_submodule_state_dict(state_dict, "decoder")
    fnd.load_state_dict(encoder_dict)
    return fnd

def get_gd(vector_size):
    gd = GraphDecoder(vector_size)
    state_dict = state_dict = torch.load(f"data/models/gae{vector_size}.pth", map_location='cpu')
    encoder_dict = extract_submodule_state_dict(state_dict, "decoder")
    gd.load_state_dict(encoder_dict)
    return gd

def get_tgd(vector_size):
    tgd = TransformerPCDecoder(vector_size, num_layers=2)
    state_dict = state_dict = torch.load(f"data/models/tgae{vector_size}.pth", map_location='cpu')
    encoder_dict = extract_submodule_state_dict(state_dict, "decoder")
    tgd.load_state_dict(encoder_dict)
    return tgd

def get_encoder(architecture, vector_size):
    encoder = None
    if vector_size not in [128, 256, 512]:
        smol.logger.error(f"Object codec supports vector sizes: 128, 256, 512. Provided: {vector_size}")
    if architecture == 'foldingnet':
        encoder = get_fne(vector_size)
    elif architecture == 'gae':
        encoder = get_ge(vector_size)
    elif architecture == 'tgae':
        encoder = get_tge(vector_size)
    else:
        smol.logger.error(f"Object codec supports architectures: foldingnet, gae, tgae. Provided: {architecture}")
    return encoder

def get_decoder(architecture, vector_size):
    decoder = None
    if vector_size not in [128, 256, 512]:
        smol.logger.error(f"Object codec supports vector sizes: 128, 256, 512. Provided: {vector_size}")
    if architecture == 'foldingnet':
        decoder = get_fnd(vector_size)
    elif architecture == 'gae':
        decoder = get_gd(vector_size)
    elif architecture == 'tgae':
        decoder = get_tgd(vector_size)
    else:
        smol.logger.error(f"Object codec supports architectures: foldingnet, gae, tgae. Provided: {architecture}")
    return decoder

import numpy as np
import open3d as o3d
import torch

def load_and_prepare_pointcloud(file_path: str, num_points: int = 2048):

    mesh_or_pc = o3d.io.read_triangle_mesh(file_path)
    if not mesh_or_pc.has_vertices():
        raise ValueError("Loaded object has no vertices")
    if isinstance(mesh_or_pc, o3d.geometry.TriangleMesh) and mesh_or_pc.has_triangles():
        mesh_or_pc.compute_vertex_normals()
        pc = mesh_or_pc.sample_points_uniformly(number_of_points=num_points)
        points = np.asarray(pc.points)
    else:
        pc = o3d.io.read_point_cloud(file_path)
        if not pc.has_points():
            raise ValueError("Loaded point cloud has no points")
        points = np.asarray(pc.points)
        if points.shape[0] > num_points:
            idx = np.random.choice(points.shape[0], num_points, replace=False)
            points = points[idx]
    points = sphere_normalization_numpy(points)
    N = points.shape[0]
    if N < num_points:
        pad = np.zeros((num_points - N, 3), dtype=points.dtype)
        points = np.vstack((points, pad))

    mask = torch.zeros(1, num_points, dtype=torch.float32)
    mask[0, :min(N, num_points)] = 1.0
    pc.points = o3d.utility.Vector3dVector(points) 
    points_tensor = torch.tensor(points.T, dtype=torch.float32).unsqueeze(0)

    return points_tensor, mask


def save_tensor_as_pcd_ply(tensor: torch.Tensor, filename: str):
    """
    Converts a 1 x 3 x N tensor to an Open3D point cloud and saves it as a .ply file.

    Args:
        tensor (torch.Tensor): Input tensor of shape (1, 3, N) on GPU or CPU.
        filename (str): Path to the output .ply file.
    """
    if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] != 3:
        raise ValueError("Tensor must have shape (1, 3, N)")

    tensor_np = tensor.squeeze(0).transpose(0, 1).detach().cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tensor_np)

    o3d.io.write_point_cloud(filename, pcd)

