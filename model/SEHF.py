import torch
from torch import nn
from spikingjelly.activation_based import neuron, surrogate, functional, layer
from model.Generator import *



class ClipperEventEncoder(nn.Module):  #input : 2015 * 448 * 280
    def __init__(self, tau:float):
        super().__init__()
        self.tau = tau
        self.snn = nn.Sequential(
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()), # tau越大衰减越慢
            layer.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            layer.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )

        functional.set_step_mode(self, step_mode='s'),

    def forward(self, x_seq): # 403*1*448*280
        # x_seq = x_seq.transpose(0, 1)  # 403*1**448*280
        x_seq = x_seq.unsqueeze(1)  # 403*1*1*448*280
        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            x = x_seq[t]
            x = self.snn(x)
            y_seq.append(x)
            del x
            torch.cuda.empty_cache()
        y = torch.cat(y_seq)  # 403*1*448*280
        del y_seq
        y = y.mean(0)  # 1*448*280
        return y
    

class eventEncoder(nn.Module):  #input : 2015 * 448 * 280
    def __init__(self, tau:float):
        super().__init__()
        self.tau = tau
        self.snn = nn.Sequential(
            neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan()), # tau越大衰减越慢
            layer.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            layer.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )

        functional.set_step_mode(self, step_mode='s'),

    def forward(self, x_seq): # 403*1*448*280
        # x_seq = x_seq.transpose(0, 1)  # 403*1**448*280
        x_seq = x_seq.unsqueeze(1)  # 403*1*1*448*280
        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            x = x_seq[t]
            x = self.snn(x)
            y_seq.append(x)
            del x
            torch.cuda.empty_cache()
        y = torch.cat(y_seq)  # 403*1*448*280
        del y_seq
        y = y.mean(0)  # 1*448*280
        return y

class REClipper(nn.Module):
    def __init__(self, output_channels:int):
        super().__init__()
        self.OnEncoder = ClipperEventEncoder(tau=10.0)
        self.OffEncoder = ClipperEventEncoder(tau=10.0)
        self.clipper = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, On_event, Off_event, rgb, device):
        On = []
        Off = []
        for i in range(len(On_event)):
            On_event_s = On_event[i]  # 403*1*448*280
            Off_event_s = Off_event[i]  # 403*1*448*280
            On_s = self.OnEncoder(On_event_s) # 1*448*280
            Off_s = self.OffEncoder(Off_event_s) # 1*448*280
            On.append(On_s.unsqueeze(0))
            Off.append(Off_s.unsqueeze(0))
            
    
        On = torch.cat(On, dim=0)   #5*1*448*280
        Off = torch.cat(Off, dim=0)
        On = On.mean(0)  # 1*448*280
        Off = Off.mean(0)
        On = On.unsqueeze(0)
        Off = Off.unsqueeze(0)
        x = torch.cat([On, Off, rgb], dim=1)
        x = self.clipper(x)
        functional.reset_net(self.OnEncoder)
        functional.reset_net(self.OffEncoder)

        return x
    

class LSTM(nn.Module):
    def __init__(self, num_layers:int):
        super().__init__()  # input shape = 3*448*280
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1), # 4*448*280
            nn.ReLU(),
            nn.Flatten(), # 1*448*280
            nn.Linear(1*448*280, 1024),
            nn.ReLU(),
            ) 
        self.lstm = nn.LSTM(input_size=1024, hidden_size=1024, num_layers=num_layers, batch_first=True)
        self.upsample = nn.Sequential(
            nn.Linear(1024, 1*448*280),
            nn.ReLU(),
            nn.Unflatten(1, (1, 448, 280)),
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x, hidden):
        x = self.conv(x)  # 1024
        lstm_output, hidden = self.lstm(x.unsqueeze(0), hidden) # 1*3*448*280
        lstm_output = self.upsample(lstm_output.squeeze(0)) # 3*448*280
        return lstm_output, hidden
    

class SEHF(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.OnEncoder = eventEncoder(tau=10.0)
        self.OffEncoder = eventEncoder(tau=10.0)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.generator = Generator(n_channels=4, bilinear=True)
        

    def forward(self, On_event, Off_event, lstm_output, clipper, device):
        On = []
        Off = []
        for i in range(len(On_event)):
            On_event_s = On_event[i].to(device, dtype=torch.float16)  # 403*1*448*280
            Off_event_s = Off_event[i].to(device, dtype=torch.float16)  # 403*1*448*280
            
            On_s = self.OnEncoder(On_event_s) # 1*448*280
            Off_s = self.OffEncoder(Off_event_s) # 1*448*280
            On.append(On_s.unsqueeze(0))
            Off.append(Off_s.unsqueeze(0))
            # del On_event_s, Off_event_s, On_s, Off_s
            # torch.cuda.empty_cache()
        functional.reset_net(self.OnEncoder)
        functional.reset_net(self.OffEncoder)
        On = torch.cat(On, dim=0)   #5*1*448*280
        Off = torch.cat(Off, dim=0)
        On = On.mean(0).unsqueeze(0) # 1*1*448*280
        Off = Off.mean(0).unsqueeze(0)    # 1*1*448*280
        x = torch.cat([On, Off], dim=1)  #1*2*448*280
        x = self.conv1(x)    # 4*448*280

        if lstm_output is None:
            prompt = self.conv3(clipper)    # 32*448*280
        else:
            prompt = torch.cat([clipper, lstm_output], dim=1)  # (7 + 3) 8*448*280
            prompt = self.conv2(prompt) # 32*448*280
        x = self.generator(x, prompt)   # 3*448*280

        return x
        

        
    
