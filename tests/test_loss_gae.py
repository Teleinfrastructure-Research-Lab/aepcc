import torch
from smol.core import smol
from loss.gae_loss import cd_gae_loss_knn_masked, GAEChamferLoss



if __name__ =="__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cd_loss_class = GAEChamferLoss().to(device)

    # Redefine base tensors
    x_base = torch.Tensor([[[1,2,3],[3,2,1],[2,1,3],[1,3,2]],[[2,3,4],[3,4,2],[2,1,3],[1,3,2]]])
    x2_base = torch.Tensor([[[11,12,13],[13,12,11],[2,1,3],[1,3,2]],[[12,13,14],[13,14,12],[2,1,3],[1,3,2]]])
    mask_base = torch.Tensor([[[1],[1],[0],[0]],[[1],[1],[1],[0]]]).long()
    # mask_base = torch.Tensor([[[1],[1],[1],[1]],[[1],[1],[1],[1]]]).long()

    num_points = 2048 
    rep_factor = num_points // 4

    x = x_base.repeat(1, rep_factor, 1).permute(0, 2, 1).to(device)
    x2 = x2_base.repeat(1, rep_factor, 1).permute(0, 2, 1).to(device)
    mask = mask_base.repeat(1, rep_factor, 1).permute(0, 2, 1).squeeze(1).to(device)

    loss1 = cd_gae_loss_knn_masked(x, x2, mask)
    loss2 = cd_loss_class.forward(x, x2, mask)

    print(f"cd func: {loss1}, cd class: {loss2}")


    x = torch.rand(32, 3, 2048) 
    mask = torch.ones(32, 2048)