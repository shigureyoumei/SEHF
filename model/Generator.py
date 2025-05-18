import torch
from torch import nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """
	#只需要输入输入的数据的通道，因为后续的通道数不会发生改变
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input（输入的图像的最后一个维度，图像的通道数）
        * `n_heads` is the number of heads in multi-head attention（多头注意力的头的个数）
        * `d_k` is the number of dimensions in each head（每个多头注意力的维度数）
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
       
        #组归一化：你有一个包含64个通道的输入，并且你设置n_groups=8，那么每个组将包含8个通道，组归一化将在这8个通道上独立地计算均值和标准差，并进行归一化
        self.norm = nn.GroupNorm(n_groups, n_channels)
        
        #将通过线性变化，将通道数增大为：多头注意力头数*每个头的维度，以及*3，用来后续划分为Q、K、V
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
     
        #输出为维度通过线性映射恢复为何输入一致
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        
  
        #首先得到输入数据的批量大小，通道维度，长，宽
        batch_size, n_channels, height, width = x.shape
       
        #将除通道数的维度进行合并，然后将通道数放在最后面
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
    
        #通过投影，将数据的维度进行提升，让其满足多头注意力的维度数
        #然后将数据的维度变化为：批量大小，像素维度（比如长*宽），头数，3*头的维度
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
   
        #将得到QKV按照最后一个维度进行维度划分，得到QKV矩阵
        q, k, v = torch.chunk(qkv, 3, dim=-1)
  
        #QK进行点积计算，维度变为：批量，像素维度，像素维度，头数
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        #将第二个维度归一化
        attn = attn.softmax(dim=2)
        #attn与v进行点积，实现加权计算，维度变为和输入的QKV一样：批量，像素大小，头数，每头维度
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        #将结果的最后两个维度合并：头数，像素大小，升维的维度
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        #将结果的维度调整为和输入一致
        res = self.output(res)
		#做残差连接
        res += x
		#将将结果的维度调整为和输入一致，将长和宽拆开
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(stride=2, kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    """Upscaling then double conv"""
 
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
 
        # if bilinear, use the normal convolutions to reduce the number of channels
        # input here is concatenation of two feature maps thus Transpose Convolution // 2
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)# //为整数除法
 
        self.conv = DoubleConv(in_channels, out_channels)
 
    def forward(self, x1, x2):
        # output = []
        # for t in range(x1.size(2)):
        #     x = x1[:, :, t, :, :]
        #     output.append(self.up(x))
        # x1 = torch.stack(output, dim=2)
        
        # input is CDHW
        # diffZ = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
 
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
                        # diffZ // 2, diffZ - diffZ // 2])
 
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
 
    def forward(self, x):
        return self.conv(x)
    

class Generator(nn.Module):
    def __init__(self, n_channels, bilinear=True): # n_channels=3
        super().__init__()
        self.n_channels = n_channels # 输入通道数
        self.bilinear = bilinear # 上采样方式
        self.inc = DoubleConv(n_channels, 64)  # n_channels=3 3*24*448*280->64*24*448*280
        self.attn = AttentionBlock(4, n_heads=4, d_k=4, n_groups=2)
        self.down1 = Down(64, 128)   # 64*24*448*280->128*24*224*140 
        self.down2 = Down(128, 256) # 128*24*224*140->256*24*112*70
        self.down3 = Down(256, 512) # 256*24*112*70->512*24*56*35
        self.down4 = Down(512, 512) # 512*24*56*35->512*24*28*17

        # self.lstm = LSTM(num_layers=3)

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 256) # 输出层

        
        
 
    def forward(self, x): # x: 2*4*280*448
        # x = self.attn(x) # 4*24*448*280
        x1 = self.inc(x) #  64*280*448

        # 四层左部分
        x2 = self.down1(x1) # 128*140*224
        x3 = self.down2(x2) # 256*70*112
        x4 = self.down3(x3) # 512*35*56
        x5 = self.down4(x4) # 512*17*28

        # lstm_out = self.lstm(x5) # 512*17*28

        # 四层右部分
        x = self.up1(x5, x4) # 512*17*28->512*35*56->CAT1024*35*56->256*35*56
        x = self.up2(x, x3) # 256*35*56->256*70*112->CAT512*70*112->128*70*112
        x = self.up3(x, x2) # 128*70*112->128*140*224->CAT256*140*224->64*140*224
        x = self.up4(x, x1) # 64*140*224->64*280*448->CAT128*280*448->64*280*448

        result = self.outc(x) # 256*280*448
        return result
    

def get_generator(n_channels=4, bilinear=True):
    return Generator(n_channels, bilinear)