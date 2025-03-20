import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)
    

class Up(nn.Module):
    """Upscaling then double conv"""
 
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
 
        # if bilinear, use the normal convolutions to reduce the number of channels
        # input here is concatenation of two feature maps thus Transpose Convolution // 2
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)# //为整数除法
 
        self.conv = DoubleConv(in_channels, out_channels)
 
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
 
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
 
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
 
    def forward(self, x):
        return self.conv(x)
    

class Generator(nn.Module):
    def __init__(self, n_channels, bilinear=True): # n_channels=32
        super().__init__()
        self.n_channels = n_channels # 输入通道数
        self.bilinear = bilinear # 上采样方式
        self.inc = DoubleConv(n_channels, 32)  # n_channels=4
        self.down1 = Down(64, 64)   # 64*224*140 原本应是(64, 128)
        self.down2 = Down(128, 128) # 128*112*70 原本应是(128, 256)
        self.down3 = Down(256, 256) # 256*56*35 原本应是(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(32, 3) # 输出层

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv4 = DoubleConv(32, 8)
        self.conv5 = DoubleConv(72, 32)
        
 
    def forward(self, x, prompt): # prompt: 32*448*280, x: 32*448*280
        x1 = self.inc(x) #  32*448*280
        x1 = torch.cat([x1, prompt], dim=1) # 32+32=64*448*280  
        prompt_copy = prompt.clone()

        # 四层左部分
        x2 = self.down1(x1) # 64*224*140
        prompt = self.conv1(prompt) # 64*224*140
        x2 = torch.cat([x2, prompt], dim=1) # 128*224*140

        x3 = self.down2(x2) # 128*112*70
        prompt = self.conv2(prompt) # 128*112*70
        x3 = torch.cat([x3, prompt], dim=1) # 256*112*70

        x4 = self.down3(x3) # 256*56*35
        prompt = self.conv3(prompt) # 256*56*35
        x4 = torch.cat([x4, prompt], dim=1) # 512*56*35

        x5 = self.down4(x4) # 512*28*17

        # 四层右部分
        x = self.up1(x5, x4) # 512*28*17->512*56*35->CAT1024*56*35->256*56*35
        x = self.up2(x, x3) # 256*56*35->256*112*70->CAT512*112*70->128*112*70
        x = self.up3(x, x2) # 128*112*70->128*224*140->CAT256*224*140->64*224*140
        x = self.up4(x, x1) # 64*224*140->64*448*280->CAT128*448*280->64*448*280

        prompt_copy = self.conv4(prompt_copy) # 8*448*280
        x = torch.cat([x, prompt_copy], dim=1) # 64+8=72*448*280
        x = self.conv5(x) # 32*448*280

        result = self.outc(x) # 最终输出
        return result