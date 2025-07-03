from torch import nn
import torch
import torch.nn as nn
from torch_scatter import scatter
import torch_geometric.nn
import os
import numpy as np

from smol.core import smol
from utils.utils import add_awgn_noise
from utils.geometry import generate_sphere_encoding, generate_grid_encoding, generate_gaussian_encoding
from utils.codec import quantize8_tensor, dequantize8_tensor, quantize_and_pack_4bit, unpack_and_dequantize_4bit, quantize_and_pack_2bit, unpack_and_dequantize_2bit, quantize_and_pack_1bit, unpack_and_dequantize_1bit
from .gcn import SimpleGraphLayer

class FoldingEncoder(nn.Module):
    def __init__(self, num_features = smol.get_config("default-params", "model", "LATENT_DIM"),
                k = smol.get_config("default-params", "model", "K_GCN")):

        super(FoldingEncoder, self).__init__()
        self.k = k
        self.n = smol.get_config("data", "NUM_POINTS")
        self.num_features = num_features
        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(num_features*2, num_features, 1),
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=False),
            nn.Conv1d(num_features, num_features, 1),
            nn.BatchNorm1d(num_features),
            nn.ReLU( inplace=False)
        )
        self.gl1 = SimpleGraphLayer(64, 128)
        self.gl2 = SimpleGraphLayer(128, num_features*2)


    def forward(self, x:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """
        :params
        x: input point cloud (B x 3 x N)
        mask: B x N - Neighbor point cloud masks

        :return
        x: codewords (B x out_channels)
        """
        batch_size, _, num_points = x.size()
        x_flatten = x.permute(0, 2, 1).reshape(-1, 3)  # (batch_size * N, )
        mask_flatten = mask.view(-1)  # (batch_size * N)
        batch_index = torch.arange(batch_size, device=x.device).repeat_interleave(num_points)

        valid_mask = mask_flatten == 1  # Boolean mask for valid nodes
        x_flatten_masked = x_flatten[valid_mask]  # Keep only nodes with mask == 1
        batch_index_masked = batch_index[valid_mask] # Keep only nodes with mask == 1
        edge_index = torch_geometric.nn.knn(x_flatten_masked, x_flatten, k=self.k, batch_x=batch_index_masked, batch_y=batch_index)  # (2, E)
        gl_idx = torch.arange(batch_size * num_points, device=x.device)[mask_flatten == 1][edge_index[1]]

        # Gather neighbors using edge index
        _, col = edge_index  # row is the target (center), col is the source (neighbor)
        neighbors = x_flatten_masked[col]  # Gather neighbor points (E, 3)
        neighbors = neighbors.view(batch_size, num_points, self.k, 3)  # (B, N, 16, 3)
        mean = torch.mean(neighbors, dim=2, keepdim=True)
        neighbors_centered = neighbors - mean  # (B, N, 16, 3)

        # Compute covariance matrices
        covariances = torch.matmul(neighbors_centered.transpose(2, 3), neighbors_centered)  # (B, N, 3, 3)
        covariances = covariances.view(batch_size, num_points, -1)  # Flatten to (B, N, 9)
        x = torch.cat([x, covariances.permute(0,2,1)], dim=1)  # (B, 12, N)

        # three layer MLP
        x = self.mlp1(x)
        x = self.gl1(x, gl_idx)
        x = self.gl2(x, gl_idx)
        #masked global max pooling
        scatter_idx = ((batch_index+1)*mask_flatten).long()
        x_feature_flatten = x.permute(0, 2, 1).reshape(-1, self.num_features*2)
        x = scatter(x_feature_flatten, scatter_idx, dim = 0, reduce = "max")[1:] # questionable practice
        x = x.view(batch_size, 1, self.num_features*2).permute(0, 2, 1)  # [B, C, N]
        # two layer MLP
        x = self.mlp2(x).squeeze(2)

        return x

class FoldingLayer(nn.Module):
    """
    The folding operation of FoldingNet
    """

    def __init__(self, in_channel: int, out_channels: list):
        super(FoldingLayer, self).__init__()

        layers = []
        for oc in out_channels[:-1]:
            conv = nn.Conv1d(in_channel, oc, 1)
            bn = nn.BatchNorm1d(oc)
            active = nn.ReLU(inplace=False)
            layers.extend([conv, bn, active])
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 1)
        layers.append(out_layer)
        
        self.layers = nn.Sequential(*layers)

    def forward(self, grids, codewords):
        """
        Parameters
        ----------
            grids: reshaped 2D grids or intermediam reconstructed point clouds
        """
        # concatenate
        x = torch.cat([grids, codewords], dim=1)
        # shared mlp
        x = self.layers(x)
        
        return x

class FoldingDecoder(nn.Module):
    """
    Decoder Module of FoldingNet
    """

    def __init__(self, in_channel=smol.get_config("default-params", "model", "LATENT_DIM"),
                pe_type = smol.get_config("default-params", "model", "DECODER_PE_TYPE"),
                R = smol.get_config("default-params", "model", "R")):
        super(FoldingDecoder, self).__init__()

        if pe_type == "sphere":
            self.pe = generate_sphere_encoding(R**2, smol.get_config("default-params", "model", "GRID_VALUES_MAX"))
            self.m = self.pe.shape[1]
            self.fold1 = FoldingLayer(in_channel + 3, [in_channel, in_channel, 3])
        elif pe_type == "gaussian":
            self.pe = generate_gaussian_encoding(R**2, std=smol.get_config("default-params", "model", "GRID_VALUES_MAX"))
            self.m = self.pe.shape[1]
            self.fold1 = FoldingLayer(in_channel + 3, [in_channel, in_channel, 3])
        else:
            self.pe = generate_grid_encoding(R, smol.get_config("default-params", "model", "GRID_VALUES_MAX"))
            self.m = self.pe.shape[1]
            self.fold1 = FoldingLayer(in_channel + 2, [in_channel, in_channel, 3])

        self.fold2 = FoldingLayer(in_channel + 3, [in_channel, in_channel, 3])

    def forward(self, x):
        """
        x: (B, C)
        """
        batch_size = x.shape[0]

        # repeat pe for batch operation
        pe = self.pe.to(x.device)                      # (2, 45 * 45)
        pe = pe.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, 45 * 45)
        
        # repeat codewords
        x = x.unsqueeze(2).repeat(1, 1, self.m)            # (B, 512, 45 * 45)
        
        # two folding operations
        recon1 = self.fold1(pe, x)
        recon2 = self.fold2(recon1, x)
        
        return recon2

@smol.register_architecture("folding-net", {
    "input_def": {
        "point_clouds":(-1, 3, smol.get_config("data", "NUM_POINTS")),
        "masks": (-1, smol.get_config("data", "NUM_POINTS"))
    },
    "output_def": {
        "z":(-1, -1),
        "out":(-1, 3, smol.get_config("default-params", "model", "R")**2)
    },
    "checkpoint-modules": ["encoder", "decoder"],
    "module_params": {
        "num_features": smol.get_config("default-params", "model", "LATENT_DIM"),
        "k": smol.get_config("default-params", "model", "K_GCN"),
        "pe_type": smol.get_config("default-params", "model", "DECODER_PE_TYPE"),
        "quantize": False,
        "quantization_depth": 16,
        "awgn": False,
        "awgn_snr": 100.0
    }
})
class AutoEncoder(nn.Module):
    def __init__(self, num_features:int = smol.get_config("default-params", "model", "LATENT_DIM"),
                 k:int = smol.get_config("default-params", "model", "K_GCN"),
                 pe_type:str = smol.get_config("default-params", "model", "DECODER_PE_TYPE"),
                 quantize:bool = False,
                 quantization_depth:int = 16,
                 awgn = False,
                 awgn_snr = 100):
        super().__init__()
        self.quantize = quantize
        self.quantization_depth = quantization_depth
        self.awgn = awgn
        self.awgn_snr = awgn_snr

        self.encoder = FoldingEncoder(num_features=num_features, k=k)
        self.decoder = FoldingDecoder(in_channel=num_features, pe_type=pe_type)
        if self.quantize == True or self.awgn == True:
            path = os.path.join(smol.get_config("data", "quantization_stats"), f"foldingnet_{num_features}.npy")
            self.quant8_stats = np.load(path)

    def forward(self, x, masks):
        z = self.encoder(x, masks)
        orig_shape = z.shape

        if self.quantize == True and self.awgn == True:
            raise ValueError("Select either AWGN or quantization!")

        if self.quantize == True:
            if self.quantization_depth == 16:
                z = z.half()
                _z = z.float()
            elif self.quantization_depth == 8:
                z = quantize8_tensor(z, self.quant8_stats)
                _z = dequantize8_tensor(z, self.quant8_stats)
            elif self.quantization_depth == 4:
                z = quantize_and_pack_4bit(z, self.quant8_stats)
                _z = unpack_and_dequantize_4bit(z, self.quant8_stats, orig_shape)
            elif self.quantization_depth == 2:
                z = quantize_and_pack_2bit(z, self.quant8_stats)
                _z = unpack_and_dequantize_2bit(z, self.quant8_stats, orig_shape)
            elif self.quantization_depth == 1:
                z = quantize_and_pack_1bit(z, self.quant8_stats)
                _z = unpack_and_dequantize_1bit(z, self.quant8_stats, orig_shape)
            else:
                raise ValueError("Only 16, 8, 4, 1 bit quantization is supported!")
        else:
            _z = z

        if self.awgn == True:
            z = add_awgn_noise(z, self.awgn_snr, self.quant8_stats)
            _z = z
        
        out = self.decoder(_z)
        return z, out