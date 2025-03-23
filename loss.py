import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import lpips  # LPIPS 需要安装 pip install lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim

# **LPIPS 计算模块**
lpips_loss_fn = lpips.LPIPS(net='vgg')  # 使用 VGG 作为特征提取网络
lpips_loss_fn = lpips_loss_fn.to('cuda' if torch.cuda.is_available() else 'cpu')


alpha = 1e-3

# **混合损失函数**
class HybridLoss(nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_ssim=0.5, lambda_lpips=0.2):
        super(HybridLoss, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.mse_loss = nn.MSELoss()

    def forward(self, gen_image, real_image):
        # **MSE 损失**
        loss_mse = self.mse_loss(gen_image, real_image)

        # **SSIM 损失（1 - SSIM）** # **LPIPS 感知损失**
        loss_ssim = 0.
        loss_lpips = 0.
        for i in range(gen_image.size(2)):
            loss_ssim += 1 - ssim(gen_image[:,:,i,:,:], real_image[:,:,i,:,:], data_range=1.0)
            loss_lpips += alpha*lpips_loss_fn(gen_image[:,:,i,:,:], real_image[:,:,i,:,:]).mean()
        loss_ssim /= gen_image.size(2)
        loss_lpips /= gen_image.size(2)

        # **最终混合损失**
        total_loss = (self.lambda_mse * loss_mse + 
                      self.lambda_ssim * loss_ssim + 
                      self.lambda_lpips * loss_lpips)

        return total_loss