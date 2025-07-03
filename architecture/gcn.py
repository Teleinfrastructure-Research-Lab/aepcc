from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

from smol.core import smol

class GraphConvPlain(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=smol.get_config("default-params", "model", "LATENT_DIM"), 
                out_channels=smol.get_config("default-params", "model", "LATENT_DIM"),
                k=smol.get_config("default-params", "model", "K_GCN"),
                num_chebyshev=smol.get_config("default-params", "model", "CHEBY_ORDER")):
        
        super(GraphConvPlain, self).__init__()
        self.k = k

        # Graph convolution layers
        self.conv1 = ChebConv(in_channels, hidden_channels, K=num_chebyshev)
        self.conv2 = ChebConv(hidden_channels, out_channels, K=num_chebyshev)

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input point cloud of shape (batch_size, numf, N)
        :param edge_index: precomputed graph conectivity index (2, E)
        :param edge_weight: precomputed graph conectivity index (E)
        :return: Output features after graph convolutions (batch_size, out_channels, N)
        """
        batch_size, numf, num_points = x.size()
        x_flatten = x.permute(0, 2, 1).reshape(-1, numf)  # (batch_size * N, )

        # Apply first graph convolution, batch norm, and ReLU
        features = self.conv1(x_flatten, edge_index, edge_weight)
        features = self.bn1(features)  # Normalize across the channel dimension
        features = self.relu(features)

        # Apply second graph convolution, batch norm, and ReLU
        features = self.conv2(features, edge_index, edge_weight)
        features = self.bn2(features)  # Normalize across the channel dimension
        features = self.relu(features)

        # Reshape back to (batch_size, out_channels, N)
        features = features.view(batch_size, num_points, -1).permute(0, 2, 1)
        return features
    
class GraphConvSingle(nn.Module):
    def __init__(self, in_channels=3, 
                out_channels=smol.get_config("default-params", "model", "LATENT_DIM"),
                k=smol.get_config("default-params", "model", "K_GCN"),
                num_chebyshev=smol.get_config("default-params", "model", "CHEBY_ORDER")):
        
        super(GraphConvSingle, self).__init__()
        self.k = k

        self.conv1 = ChebConv(in_channels, out_channels, K=num_chebyshev)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input point cloud of shape (batch_size, numf, N)
        :param edge_index: precomputed graph conectivity index (2, E)
        :param edge_weight: precomputed graph conectivity index (E)
        :return: Output features after graph convolutions (batch_size, out_channels, N)
        """
        batch_size, numf, num_points = x.size()
        x_flatten = x.permute(0, 2, 1).reshape(-1, numf)  # (batch_size * N, )

        features = self.conv1(x_flatten, edge_index, edge_weight)
        features = self.bn1(features)
        features = self.relu(features)

        # Reshape back to (batch_size, out_channels, N)
        features = features.view(batch_size, num_points, -1).permute(0, 2, 1)
        return features

class SimpleGraphLayer(nn.Module):
    """
    Graph layer.
    """
    def __init__(self, in_channel, out_channel, k=smol.get_config("default-params", "model", "K_GCN")):
        super(SimpleGraphLayer, self).__init__()
        self.k = k
        self.conv = nn.Conv1d(in_channel, out_channel, 1)

    def forward(self, x, idx):
        """
        Parameters
        ----------
            :x: tensor with size of (B, C, N)
            :param edge_index: precomputed graph connectivity index (2, E)
        """
        B, C, N = x.shape
        
        # Flatten features for easier indexing
        x_flattened = x.permute(0, 2, 1).reshape(-1, C)  # [B * N, C]

        # Gather neighbor features
        neighbors = x_flattened[idx]  # Gather neighbors via col (shape: [E, C])

        # Reshape neighbors into [B, N, k, C]
        neighbors = neighbors.view(B, N, self.k, C)

        # Max Pooling along k (neighbor) dimension
        x = neighbors.max(dim=2)[0]  # [B, N, C]

        # Reshape back to [B, C, N]
        x = x.permute(0, 2, 1)

        # Feature Map
        x = F.relu(self.conv(x))
        return x