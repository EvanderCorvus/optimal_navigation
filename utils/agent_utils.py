import torch.nn as nn

def he_initialization_leaky_relu(m, negative_slope=0.01):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=negative_slope, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
