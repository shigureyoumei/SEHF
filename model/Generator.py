import torch
from torch import nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, num_layers:int, input_channels:int):
        super().__init__()  # input shape = 512*24*28*17
        self.input_channels = input_channels
        self.num_layers = num_layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1), # 128*28*17
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1), # 32*28*17
            nn.ReLU(),
            nn.Flatten(), # 32*17*28
            nn.Linear(32*17*28, 1024),
            nn.ReLU(),
            ) 
        self.lstm = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=num_layers, batch_first=True)
        self.upsample = nn.Sequential(
            nn.Linear(1024, 32*17*28),
            nn.ReLU(),
            nn.Unflatten(1, (32, 17, 28)),
            nn.Conv2d(in_channels=32, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x_3d):   # input shape = 512*24*17*28
        hidden = None
        output = []
        for t in range(x_3d.size(2)):
            x = x_3d[:, :, t, :, :]  # 512*17*28
            x = self.conv(x)  # 1024
            out, hidden = self.lstm(x, hidden)
            x = self.upsample(out) # 512*17*28
            output.append(x)   # 512*17*28
        output = torch.stack(output, dim=2) # 512*24*17*28

        return output

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(stride=(1, 2, 2), kernel_size=(1, 2, 2)),
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
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)# //为整数除法
 
        self.conv = DoubleConv(in_channels, out_channels)
 
    def forward(self, x1, x2):
        output = []
        for t in range(x1.size(2)):
            x = x1[:, :, t, :, :]
            output.append(self.up(x))
        x1 = torch.stack(output, dim=2)
        
        # input is CDHW
        diffZ = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffY = torch.tensor([x2.size()[3] - x1.size()[3]])
        diffX = torch.tensor([x2.size()[4] - x1.size()[4]])
 
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
 
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
 
    def forward(self, x):
        return self.conv(x)
    

class Generator(nn.Module):
    def __init__(self, n_channels, bilinear=True): # n_channels=3
        super().__init__()
        self.n_channels = n_channels # 输入通道数
        self.bilinear = bilinear # 上采样方式
        self.inc = DoubleConv(n_channels, 64)  # n_channels=3 3*24*448*280->64*24*448*280
        self.down1 = Down(64, 128)   # 64*24*448*280->128*24*224*140 
        self.down2 = Down(128, 256) # 128*24*224*140->256*24*112*70
        self.down3 = Down(256, 512) # 256*24*112*70->512*24*56*35
        self.down4 = Down(512, 512) # 512*24*56*35->512*24*28*17

        self.lstm = LSTM(num_layers=3, input_channels=512)

        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 3) # 输出层

        
        
 
    def forward(self, x): # x: 4*24*448*280
        x1 = self.inc(x) #  64*24*280*448

        # 四层左部分
        x2 = self.down1(x1) # 128*24*140*224
        x3 = self.down2(x2) # 256*24*70*112
        x4 = self.down3(x3) # 512*24*35*56
        x5 = self.down4(x4) # 512*24*17*28

        lstm_out = self.lstm(x5) # 512*24*17*28

        # 四层右部分
        x = self.up1(lstm_out, x4) # 512*24*17*28->512*24*35*56->CAT1024*24*35*56->256*24*35*56
        x = self.up2(x, x3) # 256*24*35*56->256*24*70*112->CAT512*24*70*112->128*24*70*112
        x = self.up3(x, x2) # 128*24*70*112->128*24*140*224->CAT256*24*140*224->64*24*140*224
        x = self.up4(x, x1) # 64*24*140*224->64*24*280*448->CAT128*24*280*448->64*24*280*448

        result = self.outc(x) # 3*24*280*448
        return result