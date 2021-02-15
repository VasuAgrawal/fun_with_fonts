#!/usr/bin/env python3

# Modified from:
# https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/39

import progressbar
from loaders import makeLoaders
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#  channels = 3
#  transform = transforms.Compose([transforms.ToTensor(),])
#  dataset = datasets.CIFAR10(root='cifar10', train=True, download=True,transform=transform)
#  train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)

channels = 55
train_loader, test_loader = makeLoaders(1, 1, channels)

mean = torch.zeros(channels)
std = torch.zeros(channels)

for i, data in progressbar.progressbar(enumerate(train_loader)):
    data = data[0].squeeze(0) # [channels, 64, 64]
    data = data.view([-1, data.size(-2), data.size(-1)]) # make it work for 1 ch

    if (i == 0):
        size = data.size(1) * data.size(2)
    mean += data.sum((1, 2)) / size

mean /= len(train_loader)
print("mean:", mean)
mean_unsq = mean.unsqueeze(1).unsqueeze(2) # [channels, 1, 1]

for i, data in progressbar.progressbar(enumerate(train_loader)):
    data = data[0].squeeze(0)
    std += ((data - mean_unsq) ** 2).sum((1, 2)) / size

std /= len(train_loader)
std = std.sqrt()
print("std:", std)

filename = "meanstd.pt"
torch.save({"mean": mean, "std": std}, filename)
loaded = torch.load(filename)

print()
print("loaded mean:", loaded['mean'])
print("loaded std:", loaded['std'])
