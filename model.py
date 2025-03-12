import torch
from torch import nn
from spikingjelly.activation_based import neuron, surrogate, functional, layer



class eventEncoder(nn.Module):  #input : 2015 * 448 * 280
    def __init__(self, T:int ):
        super().__init__()
        self.T = T
        self.snn = nn.Sequential(
            neuron.LIFNode(tau=2, surrogate_function=surrogate.ATan()), # tau越大衰减越慢
            layer.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False),
            neuron.LIFNode(tau=2, surrogate_function=surrogate.ATan()),
            layer.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
        )

        functional.set_step_mode(self, step_mode='m'),

    def forward(self, x):
        x = x.unsqueeze(1)  # 2015*448*280 -> 2015*1*448*280
        x = self.snn(x)
        x = x.mean(0)   # 1*448*280
        return x
