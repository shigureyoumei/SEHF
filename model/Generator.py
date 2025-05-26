import torch
from torch import nn
import torch.nn.functional as F
import model.transformer as tf

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
        self.inc = DoubleConv(n_channels, 64)  # n_channels=3 3*24*224*140->64*24*224*140
        self.down1 = Down(64, 128)   # 64*24*224*140->128*24*112*70
        self.down2 = Down(128, 256) # 128*24*112*70->256*24*56*35
        self.down3 = Down(256, 512) # 256*24*56*35->512*24*28*18
        self.down4 = Down(512, 512) # 512*24*28*18->512*24*14*9

        # self.lstm = LSTM(num_layers=3)

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 3) # 输出层

        self.colorInsert = tf.get_transformer(patch_size=4, depth=1, dim=64, heads=16, mlp_dim=128, dim_head=8, dropout=0.1, emb_dropout=0.1)

        
        
 
    def forward(self, x): # x: 2*4*280*448

        x = self.colorInsert(x)
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