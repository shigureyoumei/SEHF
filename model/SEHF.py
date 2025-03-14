import torch
from torch import nn
from spikingjelly.activation_based import neuron, surrogate, functional, layer
from Generator import *



class eventEncoder(nn.Module):  #input : 2015 * 448 * 280
    def __init__(self, tau:float):
        super().__init__()
        self.tau = tau
        self.snn = nn.Sequential(
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()), # tau越大衰减越慢
            layer.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()),
            layer.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
        )

        functional.set_step_mode(self, step_mode='m'),

    def forward(self, x):
        x = x.unsqueeze(1)  # 2015*448*280 -> 2015*1*448*280
        x = self.snn(x)
        x = x.mean(0)   # 1*448*280
        return x

class REClipper(nn.Module):
    def __init__(self, output_channels:int):
        super().__init__()
        self.output_channels = output_channels
        self.clipper = nn.Sequential(
            nn.Conv2d(input_channels=5, output_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(input_channels=16, output_channels=output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.clipper(x)
        return x
    

class LSTM(nn.Module):
    def __init__(self, num_layers:int):
        super().__init__()  # input shape = 3*448*280
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels=3, output_channels=4, kernel_size=3, stride=1, padding=1), # 4*448*280
            nn.ReLU(),
            nn.Flatten()) # 4*448*280
        self.lstm = nn.LSTM(input_size=4*448*280, hidden_size=3*448*280, num_layers=num_layers, batch_first=True)

    def forward(self, x, hidden):
        x = self.conv(x)  # 4*448*280
        lstm_output, hidden = self.lstm(x.unsqueeze(0), hidden) # 1*3*448*280
        lstm_output = lstm_output.reshape(-1, 3, 448, 280)  # 3*448*280
        return lstm_output, hidden
    

class SEHF(nn.Module):
    def __init__(self, first:bool, clipper=None):
        super().__init__()
        self.first = first
        self.clipper = clipper

        self.OnEncoder = eventEncoder(tau=10)
        self.OffEncoder = eventEncoder(tau=10)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels=2, output_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(input_channels=12, output_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.generator = Generator(n_channels=16, bilinear=True)
        

    def forward(self, On_event, Off_event, lstm_output):
        On = self.OnEncoder(On_event)  # 1*448*280
        Off = self.OffEncoder(Off_event)    # 1*448*280
        x = torch.cat([On, Off], dim=1)
        x = self.conv1(x)    # 4*448*280
        prompt = torch.cat([self.clipper, lstm_output], dim=1)  # (9 + 3) 12*448*280
        prompt = self.conv2(prompt) # 32*448*280
        x = self.generator(x, prompt)   # 3*448*280

        return x
        

        
    
