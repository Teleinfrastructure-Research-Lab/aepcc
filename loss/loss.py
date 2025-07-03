import torch
from torch_geometric.nn import knn
from torch_scatter import scatter

from smol.core import smol

@smol.register_loss("cd-loss-tgae", {
    "input_def": {
        "point_clouds": (-1, 3, smol.get_config("data", "NUM_POINTS")),
        "out": (-1, 3, smol.get_config("data", "NUM_POINTS")),
        "masks": (-1, smol.get_config("data", "NUM_POINTS"))
    },
    "output_def": {
        "cd":()
    },
    "backward_keys": ["cd"],
    "module_params": {
    }
})
@smol.register_loss("cd-loss", {
    "input_def": {
        "point_clouds": (-1, 3, smol.get_config("data", "NUM_POINTS")),
        "out": (-1, 3, smol.get_config("default-params", "model", "R")**2),
        "masks": (-1, smol.get_config("data", "NUM_POINTS"))
    },
    "output_def": {
        "cd":()
    },
    "backward_keys": ["cd"],
    "module_params": {
    }
})
class ChamferLoss(torch.nn.Module):
    """
    Chamfer distance loss class
    """

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x:torch.Tensor, y:torch.Tensor)->torch.Tensor:
        """
        Compute pairwise distances between points in x and y.
        
        Args:
            x: Tensor of shape (B, 3, N)
            y: Tensor of shape (B, 3, M)
        
        Returns:
            Pairwise distance matrix of shape (B, N, M)
        """
        # Transpose inputs to shape (B, N, 3)
        x = x.transpose(1, 2)  # (B, N, 3)
        y = y.transpose(1, 2)  # (B, M, 3)
        
        # Get sizes
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        
        # Compute pairwise distance
        xx = x.pow(2).sum(dim=-1)  # (B, N)
        yy = y.pow(2).sum(dim=-1)  # (B, M)
        zz = torch.bmm(x, y.transpose(2, 1))  # (B, N, M)
        rx = xx.unsqueeze(2).expand(bs, num_points_x, num_points_y)  # (B, N, M)
        ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y)  # (B, N, M)
        P = rx + ry - 2 * zz  # Pairwise distance matrix
        
        return P

    def forward(self, x:torch.Tensor, y:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """
        Compute the Chamfer distance between two point clouds using k-NN and a mask.
    
        Parameters:
        x (Tensor): Label point cloud of shape (B, C, N)
        y (Tensor): Predicted point cloud of shape (B, C, M)
        mask (Tensor): Mask of shape (B, N) indicating valid points in x
        """
        B, _, N = x.size()
        _, _, M = y.size()

        mask_flatten = mask.view(-1)  # (batch_size * N)
        batch_index = torch.arange(B, device=x.device).repeat_interleave(N)
        scatter_idx = ((batch_index+1)*mask_flatten).long()
        P = self.batch_pairwise_dist(x, y)
        P_flatten = P.reshape(B*N, M)
        mins = scatter(P_flatten, scatter_idx, dim = 0, reduce = "min")[1:]
        loss_1 = torch.mean(mins)
        mins, _ = torch.min(P_flatten, 1)
        loss_2 = torch.mean(mins[mask_flatten == 1])
        return loss_1 + loss_2

def cd_loss_knn_masked(x:torch.Tensor, y:torch.Tensor, mask:torch.Tensor, squared=True):
    """
    Compute the Chamfer distance between two point clouds using k-NN and a mask.
    
    Parameters:
    x (Tensor): Label point cloud of shape (B, C, N)
    y (Tensor): Predicted point cloud of shape (B, C, M)
    mask (Tensor): Mask of shape (B, N) indicating valid points in x
    squared (bool): Whether to use squared distances (default: True)
    
    Returns:
    loss (Tensor): The Chamfer distance loss
    """
    B, _, N = x.size()
    _, _, M = y.size()
    
    mask_flatten = mask.view(-1)
    x_flatten = x.permute(0, 2, 1).reshape(-1, 3)
    y_flatten = y.permute(0, 2, 1).reshape(-1, 3)
    batch_index_x = torch.arange(B, device=x.device).repeat_interleave(N)
    batch_index_y = torch.arange(B, device=x.device).repeat_interleave(M)
    x_flatten_masked = x_flatten[mask_flatten == 1]
    batch_index_x_masked = batch_index_x[mask_flatten == 1]
    edge_index_xy = knn(x_flatten_masked, y_flatten, k=1, batch_x=batch_index_x_masked, batch_y=batch_index_y)
    edge_index_yx = knn(y_flatten, x_flatten_masked, k=1, batch_x=batch_index_y, batch_y=batch_index_x_masked)
    y_nn = x_flatten_masked[edge_index_xy[1]]
    x_nn = y_flatten[edge_index_yx[1]]

    if squared:
        dist_x_y = torch.sum((x_flatten_masked - x_nn) ** 2, dim=1)
        dist_y_x = torch.sum((y_flatten - y_nn) ** 2, dim=1)
    else:
        dist_x_y = torch.norm(x_flatten_masked - x_nn, dim=1, p=2)
        dist_y_x = torch.norm(y_flatten - y_nn, dim=1, p=2)
    
    loss = dist_x_y.mean() + dist_y_x.mean()

    return loss