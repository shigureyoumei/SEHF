import torch
from torch import nn
from model.Generator import *
    


# all inputs and outputs omit batch size

class REClipper(nn.Module):
    def __init__(self, output_channels:int):
        super().__init__()
        
        self.clipper = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, event, rgb):
        x = torch.cat([event, rgb], dim=1)  # 4*280*448
        x = self.clipper(x) # 2*280*448
        x = x.unsqueeze(2).repeat(1, 1, 24, 1, 1)  # 2*24*280*448
        return x
    

    

class SEHF(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.REClipper = REClipper(output_channels=2)
        self.generator = Generator(n_channels=3, bilinear=True)
        

    def forward(self, event_first, rgb_first, event_input):
        clipper = self.REClipper(event_first, rgb_first)  # 2*24*280*448
        event_input = torch.cat([event_input, clipper], dim=1)  # 3*24*448*280

        x = self.generator(event_input)  # 3*24*448*280

        return x
        

        
    
