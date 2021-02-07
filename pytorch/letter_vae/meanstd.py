#!/usr/bin/env python3

# https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/39

from loaders import makeLoaders
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

#  transform = transforms.Compose([transforms.ToTensor(),])
#  
#  dataset = datasets.CIFAR10(root='cifar10', train=True, download=True,transform=transform)
#  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
#  
#  mean = torch.zeros(3)
#  std = torch.zeros(3)
#  
#  for i, data in enumerate(dataloader):
#      if (i % 10000 == 0): print(i)
#      data = data[0].squeeze(0)
#      print(data.size())
#      break
#      if (i == 0): size = data.size(1) * data.size(2)
#      mean += data.sum((1, 2)) / size
#  
#  mean /= len(dataloader)
#  print(mean)
#  mean = mean.unsqueeze(1).unsqueeze(2)
#  
#  for i, data in enumerate(dataloader):
#      if (i % 10000 == 0): print(i)
#      data = data[0].squeeze(0)
#      std += ((data - mean) ** 2).sum((1, 2)) / size
#  
#  std /= len(dataloader)
#  std = std.sqrt()
#  print(std)
#  

train_loader, test_loader = makeLoaders(1, 1)

mean = torch.zeros(1)
std = torch.zeros(1)

for i, data in enumerate(train_loader):
    if (i % 10000 == 0):
        print(i)
    data = data[0].squeeze(0)
    if (i == 0):
        size = data.size(0) * data.size(1)
    mean += data.sum((0, 1)) / size

mean /= len(train_loader)
print(mean)
mean = mean.unsqueeze(0).unsqueeze(1)
mean = mean.view(1).item()

for i, data in enumerate(train_loader):
    if (i % 10000 == 0): print(i)
    data = data[0].squeeze(0)
    std += ((data - mean) ** 2).sum((0, 1)) / size

std /= len(train_loader)
std = std.sqrt()
print(std)
