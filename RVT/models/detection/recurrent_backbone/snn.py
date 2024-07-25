import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, layer, surrogate

class SpkEnc(nn.Module):
    def __init__(
            self, 
            in_channels=3, 
            out_channels=128, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            tau=2, 
            decay_input=True, 
            v_threshold=1, 
            v_reset=0, 
            surrogate_function=None, 
            detach_reset=False, 
            step_mode='m', 
            backend='torch', 
            store_v_seq=False, 
        ):
        super().__init__()
        
        self.step_mode = step_mode
        
        self.conv1 = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                  kernel_size=kernel_size, stride=stride, padding=padding, 
                                  step_mode=step_mode)
        self.lif1  = neuron.LIFNode(surrogate_function=surrogate.ATan(), 
                                    step_mode=step_mode)
        
    def forward(self, x):
        # x: [L, B, C, H, W]
        x = self.conv1(x)
        x = self.lif1(x)
        
        return x