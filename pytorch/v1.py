#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, 16, 3, padding=1)


    def forward(self, x):
        x = self.conv1(x)
        print("after conv1", x.shape)
        x = F.max_pool2d(F.relu(x), (2, 2))
        print("after maxpool", x.shape)
        x = self.conv2(x)
        print("after conv2", x.shape)

        return x


net = Net()
print(net)

i = torch.randn(1, 1, 32, 32)
net.forward(i)
