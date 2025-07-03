from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import torch_geometric.nn
import os
import numpy as np

from smol.core import smol
from utils.utils import add_awgn_noise
from utils.codec import quantize8_tensor, dequantize8_tensor, quantize_and_pack_4bit, unpack_and_dequantize_4bit, quantize_and_pack_2bit, unpack_and_dequantize_2bit, quantize_and_pack_1bit, unpack_and_dequantize_1bit
from .gcn import GraphConvSingle
from .gae import InvertedFiLMPositionalEncoding

class TransformerPCEncoder(nn.Module):
    """
    Transformer based encoder.
    """
    def __init__(self, num_features = smol.get_config("default-params", "model", "LATENT_DIM"),
                class_num = smol.get_config("data", "CLASS_NUM"),
                k = smol.get_config("default-params", "model", "K_GCN"),
                num_heads:int = smol.get_config("default-params", "model", "TE_HEADS"),
                num_layers:int = smol.get_config("default-params", "model", "TE_LAYERS"),
                film_k:float = smol.get_config("default-params", "model", "FILM_K"),
                film_flag:bool = True,
                hyper_sphere:bool = True):
        super(TransformerPCEncoder, self).__init__()

        self.class_num = class_num
        self.num_features = num_features
        self.k = k
        self.film_k = film_k
        self.film_flag = film_flag
        self.hyper_sphere = hyper_sphere

        self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False)
        )

        self.tel = nn.TransformerEncoderLayer(d_model=num_features*2, 
                                           nhead=num_heads, 
                                           dim_feedforward=num_features*2, 
                                           dropout=0.1,
                                           batch_first=True)
        self.te = nn.TransformerEncoder(self.tel, num_layers=num_layers)

        self.pe = InvertedFiLMPositionalEncoding(num_features*2, hidden_dim=num_features*2)

        if film_flag:
            self.map_class = nn.Sequential(
                nn.Conv1d(class_num, num_features, 1),
                nn.BatchNorm1d(num_features),
                nn.ReLU( inplace=False),
                )
            self.gamma = nn.Sequential(
                nn.Conv1d(num_features, num_features*2, 1),
                nn.BatchNorm1d(num_features*2),
                nn.ReLU(inplace=False),
                )
            self.betta = nn.Sequential(
                nn.Conv1d(num_features, num_features*2, 1),
                nn.BatchNorm1d(num_features*2),
                nn.ReLU(inplace=False),
                )

        self.graph_layer1 = GraphConvSingle(in_channels=64, out_channels=num_features*2, k=self.k)
        self.fc = nn.Linear(in_features=num_features*2, out_features=num_features, bias=True)


    def forward(self, x:torch.Tensor, clss:torch.Tensor, mask:torch.Tensor):
        """
        :params
        x: input point cloud (B x 3 x N)
        clss: B x class_num x 1 - one-hot encoding of object classes
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
        gl_idx = torch.stack([edge_index[0],
                              torch.arange(batch_size * num_points, device=x.device)[mask_flatten == 1][edge_index[1]]])

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

        # two layer MLP
        x = self.mlp1(x)

        # graph layer
        edge_weights = torch.ones_like(gl_idx[0],  dtype=torch.float)
        x = self.graph_layer1(x, gl_idx, edge_weights)

        # Apply PE and go trough transformer
        x = self.pe(x)
        x = x.permute(0, 2, 1)
        mask_te = mask != 1
        x = self.te(x, src_key_padding_mask = mask_te)
        x = x.permute(0, 2, 1)

        #masked global max pooling
        scatter_idx = ((batch_index+1)*mask_flatten).long()
        x_feature_flatten = x.permute(0, 2, 1).reshape(-1, self.num_features*2)
        x = scatter(x_feature_flatten, scatter_idx, dim = 0, reduce = "max")[1:] # questionable practice
        x = x.view(batch_size, 1, self.num_features*2).permute(0, 2, 1)  # [B, C, N]

        if self.film_flag:
            clss_embeddings = self.map_class(clss)
            x = self.film_k*(self.gamma(clss_embeddings)*x + self.betta(clss_embeddings)) + (1-self.film_k)*x
        
        x = self.fc(x.squeeze(2))

        # Enforce hyperspherical embedding
        if self.hyper_sphere:
            x = F.normalize(x, p=2, dim=1)

        return x
    
class TransformerPCDecoder(nn.Module):
    """
    Transformer based decoder.
    """
    def __init__(self, num_features = smol.get_config("default-params", "model", "LATENT_DIM"),
                k = smol.get_config("default-params", "model", "K_GCN"),
                num_points = smol.get_config("data", "NUM_POINTS"),
                num_heads:int = smol.get_config("default-params", "model", "TE_HEADS"),
                num_layers:int = smol.get_config("default-params", "model", "TE_LAYERS"),):
        super(TransformerPCDecoder, self).__init__()

        self.num_features = num_features
        self.k = k
        self.num_points = num_points

        self.mlp2 = nn.Sequential(
            nn.Conv1d(num_features*2, num_features, 1),
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=False),
            nn.Conv1d(num_features, 3, 1)
        )

        self.pe = InvertedFiLMPositionalEncoding(num_features*2, hidden_dim=num_features*2)
        self.tel = nn.TransformerEncoderLayer(d_model=num_features*2, 
                                           nhead=num_heads, 
                                           dim_feedforward=num_features*2, 
                                           dropout=0.1,
                                           batch_first=True)
        self.te = nn.TransformerEncoder(self.tel, num_layers=num_layers)

        self.fc = nn.Linear(in_features=num_features, out_features=num_features*2, bias=True)


    def forward(self, x:torch.Tensor):
        """
        x: (B, C)
        """
        x = self.fc(x).unsqueeze(2)
        x = x.expand(-1,-1, self.num_points)
        x = self.pe(x)
        x = x.permute(0, 2, 1)
        x = self.te(x)
        x = x.permute(0, 2, 1)
        x = self.mlp2(x)

        return x

@smol.register_architecture("tgae", {
    "input_def": {
        "point_clouds":(-1, 3, smol.get_config("data", "NUM_POINTS")),
        "cls_onehots":(-1, smol.get_config("data", "CLASS_NUM"), 1),
        "masks": (-1, smol.get_config("data", "NUM_POINTS"))
    },
    "output_def": {
        "out":(-1, 3, smol.get_config("data", "NUM_POINTS")),
        "z":(-1, -1),
    },
    "checkpoint-modules": ["encoder", "decoder"],
    "module_params": {
        "num_features": smol.get_config("default-params", "model", "LATENT_DIM"),
        "class_num": smol.get_config("data", "CLASS_NUM"), 
        "k": smol.get_config("default-params", "model", "K_GCN"),
        "film_k": smol.get_config("default-params", "model", "FILM_K"),
        "film_flag": True,
        "hyper_sphere": True,
        "num_points": smol.get_config("data", "NUM_POINTS"),
        "num_heads": smol.get_config("default-params", "model", "TE_HEADS"),
        "num_layers_enc": smol.get_config("default-params", "model", "TE_LAYERS"),
        "num_layers_dec": smol.get_config("default-params", "model", "TE_LAYERS"),
        "quantize": False,
        "quantization_depth": 16,
        "awgn": False,
        "awgn_snr": 100.0
    }
})
class TransformerGraphAutoencoder(nn.Module):


    def __init__(self, num_features = smol.get_config("default-params", "model", "LATENT_DIM"),
                class_num = smol.get_config("data", "CLASS_NUM"),
                k = smol.get_config("default-params", "model", "K_GCN"),
                film_k:float = smol.get_config("default-params", "model", "FILM_K"),
                film_flag:bool = True,
                hyper_sphere:bool = True,
                num_points = smol.get_config("data", "NUM_POINTS"),
                num_heads = smol.get_config("default-params", "model", "TE_HEADS"),
                num_layers_enc = smol.get_config("default-params", "model", "TE_LAYERS"),
                num_layers_dec = smol.get_config("default-params", "model", "TE_LAYERS"),
                quantize = False,
                quantization_depth:int = 16,
                awgn = False,
                awgn_snr = 100):
        super(TransformerGraphAutoencoder, self).__init__()
        self.num_features = num_features
        self.k = k
        self.class_num = class_num
        self.film_k = film_k
        self.film_flag = film_flag
        self.hyper_sphere = hyper_sphere
        self.num_points = num_points
        self.num_heads = num_heads
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.quantize = quantize
        self.quantization_depth = quantization_depth
        self.awgn = awgn
        self.awgn_snr = awgn_snr

        self.encoder = TransformerPCEncoder(num_features, class_num, k, num_heads, 
                                            num_layers_enc, film_k, film_flag, hyper_sphere)
        self.decoder = TransformerPCDecoder(num_features, k, num_points, num_heads, num_layers_dec)
        if self.quantize == True or self.awgn == True:
            path = os.path.join(smol.get_config("data", "quantization_stats"), f"tgae_{num_features}.npy")
            self.quant8_stats = np.load(path)

    def forward(self, x:torch.Tensor, clss:torch.Tensor, mask:torch.Tensor):

        z = self.encoder(x, clss, mask)
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
        return out, z