import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Downsampling layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 8, 1)
        self.fc1 = nn.Linear(128, 30)

        # Upsampling layers
        self.fc2 = nn.Linear(30, 128)
        self.conv5 = nn.Conv2d(8, 16, 1)
        self.upsample1 = nn.Upsample((8, 8), mode="bilinear")
        self.conv6 = nn.Conv2d(16, 32, 3, padding=1)
        self.upsample2 = nn.Upsample((16, 16), mode="bilinear")
        self.conv7 = nn.Conv2d(32, 32, 3, padding=1)
        self.upsample3 = nn.Upsample((32, 32), mode="bilinear")
        self.conv8 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        # Downsample

        # 1 x 28 x 28 input
        x = F.relu(self.conv1(x))  # 32 x 32 x 32 output
        x = F.max_pool2d(x, 2)  # 32 x 16 x 16 output
        x = F.relu(self.conv2(x))  # 32 x 16 x 16 output
        x = F.max_pool2d(x, 2)  # 32 x 8 x 8 output
        x = F.relu(self.conv3(x))  # 16 x 8 x 8 output
        x = F.max_pool2d(x, 2)  # 16 x 4 x 4 output
        x = F.relu(self.conv4(x))  # 8 x 4 x 4 output
        x = torch.flatten(x, 1)  # 1 x 128 output
        x = self.fc1(x)  # 1 x 10 output

        # And upsample
        x = self.fc2(x)  # 1 x 128 output
        x = x.view(-1, 8, 4, 4)  # 8 x 4 x 4 output
        x = F.relu(self.conv5(x))  # 16 x 4 x 4 output
        x = self.upsample1(x)  # 16 x 8 x 8 output
        x = F.relu(self.conv6(x))  # 32 x 8 x 8 output
        x = self.upsample2(x)  # 32 x 16 x 16 output
        x = F.relu(self.conv7(x))  # 32 x 16 x 16 output
        x = self.upsample3(x)  # 32 x 32 x 32 output
        x = F.relu(self.conv8(x))  # 1 x 32 x 32 output

        # 1 x 28 x 28 output
        return x[:, :, 2:-2, 2:-2]

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x, x_hat)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001, weight_decay=0.0001
        )
        return optimizer

    def train_dataloader(self):

        whiten_mnist = transforms.Compose(
            [transforms.ToTensor(),
                #  transforms.Normalize((0.1307,), (0.3081,))
                ]
        )
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "/data/datasets",
                train=True,
                download=True,
                transform=whiten_mnist,
            ),
            batch_size=64,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )
        return train_loader

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = nn.functional.mse_loss(x, x_hat)
        tensorboard_logs = {"val_loss": loss}

        if batch_idx == 0:
            self.logger.experiment.add_images(
                "validation_input", x, self.global_step
            )
            self.logger.experiment.add_images(
                "validation_output", x_hat, self.global_step
            )

        return {"loss": loss, "log": tensorboard_logs}

    def val_dataloader(self):

        whiten_mnist = transforms.Compose(
            [transforms.ToTensor(),
                #  transforms.Normalize((0.1307,), (0.3081,))
                ]
        )
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "/data/datasets",
                train=False,
                download=True,
                transform=whiten_mnist,
            ),
            batch_size=64,
            shuffle=False,
            pin_memory=True,
            num_workers=4,
        )
        return val_loader


def main():
    pl.seed_everything(0)
    model = Model()

    trainer = pl.Trainer(gpus=1, num_nodes=1)
    trainer.fit(model)


if __name__ == "__main__":
    main()
