import torch
from smol.core import smol
from loss.loss import cd_loss_knn_masked, ChamferLoss
# from models.loss import cd_loss_kaolin


if __name__ =="__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cd_loss_class = ChamferLoss().to(device)

    # Redefine base tensors
    x_base = torch.Tensor([[[1,2,3],[3,2,1],[2,1,3],[1,3,2]],[[2,3,4],[3,4,2],[2,1,3],[1,3,2]]])
    x2_base = torch.Tensor([[[11,12,13],[13,12,11],[2,1,3],[1,3,2]],[[12,13,14],[13,14,12],[2,1,3],[1,3,2]]])
    mask_base = torch.Tensor([[[1],[1],[0],[0]],[[1],[1],[1],[0]]]).long()
    # mask_base = torch.Tensor([[[1],[1],[1],[1]],[[1],[1],[1],[1]]]).long()

    # Specify the number of points
    num_points = 2048  # Change this value as needed
    rep_factor = num_points // 4  # Since base has 4 points

    # Expand tensors
    x = x_base.repeat(1, rep_factor, 1).permute(0, 2, 1).to(device)
    x2 = x2_base.repeat(1, rep_factor, 1).permute(0, 2, 1).to(device)
    mask = mask_base.repeat(1, rep_factor, 1).permute(0, 2, 1).squeeze(1).to(device)

    loss1 = cd_loss_knn_masked(x, x2, mask)
    # loss2 = cd_loss_kaolin(x, x2)
    loss2 = cd_loss_class.forward(x, x2, mask)

    print(f"new: {loss1}, kaolin/ChamferLoss: {loss2}")


    #test cd_loss_knn_masked2
    x = torch.rand(32, 3, 2048) 
    mask = torch.ones(32, 2048)