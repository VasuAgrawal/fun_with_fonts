import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 4, 1)
        self.fc1 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x


    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_epoch(model, device, loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        #  print(f"Train batch {batch_idx}: {loss.item()}")


def test(model, device, loader):
    model.eval()
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim = 1, keepdim=True) # index of max prob
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /=  len(loader.dataset)
    print(f"Test: loss {test_loss}, got {correct} / {len(loader.dataset)} right")




def main():
    torch.manual_seed(42)

    device = torch.device("cuda:0")
    model = Net().to(device)

    print(f"Model has {model.param_count()} trainable parameters")

  
    whiten_mnist = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST("/data/datasets",
        train=True, download=True, transform=whiten_mnist), 
        batch_size = 64, shuffle=True, pin_memory=True )
    test_loader = torch.utils.data.DataLoader(datasets.MNIST("/data/datasets",
        train=False, download=True, transform=whiten_mnist),
        batch_size = 10000, shuffle=True, pin_memory=True )

    optimizer = torch.optim.Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size = 1, gamma = 0.7)
    for i, epoch in enumerate(range(1, 5)):
        train_epoch(model, device, train_loader, optimizer)
        test(model, device, test_loader)
        scheduler.step()

        torch.save(model.state_dict(), f"mnist_{i}.pt")


if __name__ == "__main__":
    main()
