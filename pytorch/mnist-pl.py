import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pytorch_lightning as pl

class Net(pl.LightningModule):
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


    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.nll_loss(self(x), y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss" : loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


def main():

    whiten_mnist = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "/data/datasets", train=True, download=True, transform=whiten_mnist
        ),
        batch_size=64,
        shuffle=True,
        pin_memory=True,
    )

    model = Net()
    trainer = pl.Trainer(gpus=2)
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
