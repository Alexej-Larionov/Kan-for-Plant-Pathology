import torch
from torch import nn
from convkan import ConvKAN, LayerNorm2D

device="cuda:0" if torch.cuda.is_available() else "cpu"

#--------------------MODEL------------------------
def create_model():
    model=nn.Sequential(
        ConvKAN(3,32,padding=1, kernel_size=3, stride=1,spline_order = 3, grid_size=5, grid_range=(-10,10)),
        ConvKAN(32,64,padding=1, kernel_size=3, stride=1,spline_order = 1, grid_size=5, grid_range=(-10,10)),
        LayerNorm2D(64),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
    ).to(device)
    return model
