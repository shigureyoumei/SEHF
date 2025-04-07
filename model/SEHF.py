import torch
from torch import nn
from model.Generator import *
from torchvision.models import resnet18
    


# all inputs and outputs omit batch size

class REClipper(nn.Module):
    def __init__(self, output_channels:int):
        super().__init__()
        
        # self.clipper = nn.Sequential(
        #     nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=8, out_channels=output_channels, kernel_size=3, padding=1),
        #     nn.ReLU(),
        # )
        self.resnet = resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # remove the last two layers
        
        self.upsample = nn.Upsample(size=(280, 448), mode='bilinear', align_corners=True)
        self.clipper = nn.Conv2d(in_channels=512, out_channels=output_channels, kernel_size=3, padding=1)

    def forward(self, event, rgb):
        x = torch.cat([event, rgb], dim=1)  # 5*280*448
        # x = self.clipper(x) # 2*280*448

        x = self.resnet(x)  # 512*280*448
        x = self.upsample(x)
        x = self.clipper(x)

        x = x.unsqueeze(2).repeat(1, 1, 24, 1, 1)  # 2*24*280*448
        return x
    

    

class SEHF(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.REClipper = REClipper(output_channels=1)
        self.generator = Generator(n_channels=3, bilinear=True)
        # self.manifier = nn.parameter.Parameter(torch.tensor([10.0], requires_grad=True))
        

    def forward(self, event_first, rgb_first, event_input):
        clipper = self.REClipper(event_first, rgb_first)  # 1*24*280*448
        event_input = torch.cat([event_input, clipper], dim=1)  # 3*24*448*280

        x = self.generator(event_input)  # 3*24*448*280

        return x
        

        
    
