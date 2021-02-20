import argparse
import pytorch_lightning as pl
parser = argparse.ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser.add_argument(
    "-n",
    "--channels",
    default=1,
    type=int,
    help="Number of channels",
)
parser.add_argument(
    "-z",
    "--hidden",
    default=16,
    type=int,
    help="Number of latent dimensions",
)
parser.add_argument(
    "-c",
    "--comment",
    default="autoencoder",
    help="Comment to append to folder name",
)

args = parser.parse_args()


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import loaders

def flattenChannels(stacked):
    N, C, H, W = stacked.shape
    channels = [stacked[:, c, :, :].view(N, 1, H, W) for c in range(C)]
    return torch.cat(channels, 2)

class Autoencoder(pl.LightningModule):
    RELU_LEAK = 5 ** 0.5
    CHANNELS1 = 16
    CHANNELS2 = 32
    CHANNELS3 = 64
    CHANNELS4 = 128
    CHANNELS5 = 128
    CHANNELS6 = 128

    DECODER_FC1 = 16
    DECODER_FC2 = 128
    DECODER_FC3 = 256

    def __init__(self, hidden, input_channels):
        super().__init__()
        self.lr = 0.0001
        self.save_hyperparameters()
        loaded = torch.load(
           "/data/datasets/fonts/rendered/ocr_line_split_05_val/64/meanstd.pt")
        self.input_mean = loaded['mean'][:input_channels]
        self.input_std = loaded['std'][:input_channels]
        #  print("Input mean:", self.input_mean)
        #  print("Input std: ", self.input_std)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, self.CHANNELS1, 4, stride=2, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),

            nn.Conv2d(self.CHANNELS1, self.CHANNELS2, 4, stride=2, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),

            nn.Conv2d(self.CHANNELS2, self.CHANNELS3, 4, stride=2, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),

            nn.Conv2d(self.CHANNELS3, self.CHANNELS4, 4, stride=2, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),

            nn.Conv2d(self.CHANNELS4, self.CHANNELS5, 4, stride=2, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
        )

        self.encoder_linear = nn.Sequential(
            nn.Linear(self.CHANNELS5 * 2 * 2, self.CHANNELS6),
            nn.LeakyReLU(self.RELU_LEAK),
        )

        self.encoder_mu = nn.Sequential(
            nn.Linear(
                self.CHANNELS6, hidden
            ),  # No activation, let it go wherever
        )

        self.encoder_log_sigma = nn.Sequential(
            nn.Linear(self.CHANNELS6, hidden),  # Will be exponentiated later
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(hidden, self.CHANNELS6),
            nn.LeakyReLU(self.RELU_LEAK),

            nn.Linear(self.CHANNELS6, self.CHANNELS5 * 2 * 2),
            nn.LeakyReLU(self.RELU_LEAK),
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(
                self.CHANNELS5, self.CHANNELS4, 4, stride=2, padding=1
            ),
            nn.LeakyReLU(self.RELU_LEAK),

            nn.ConvTranspose2d(
                self.CHANNELS4, self.CHANNELS3, 4, stride=2, padding=1
            ),
            nn.LeakyReLU(self.RELU_LEAK),

            nn.ConvTranspose2d(
                self.CHANNELS3, self.CHANNELS2, 4, stride=2, padding=1
            ),
            nn.LeakyReLU(self.RELU_LEAK),

            nn.ConvTranspose2d(
                self.CHANNELS2, self.CHANNELS1, 4, stride=2, padding=1
            ),
            nn.LeakyReLU(self.RELU_LEAK),

            nn.ConvTranspose2d(
                self.CHANNELS1, self.DECODER_FC1, 4, stride=2, padding=1
            ),
            nn.LeakyReLU(self.RELU_LEAK),

            # Begin "fully connected layers"
            nn.Conv2d(self.DECODER_FC1, self.DECODER_FC2, 1),
            nn.LeakyReLU(self.RELU_LEAK),

            nn.Conv2d(self.DECODER_FC2, self.DECODER_FC3, 1),
            nn.LeakyReLU(self.RELU_LEAK),

            nn.Conv2d(self.DECODER_FC3, input_channels, 1),
            #  nn.Sigmoid(),  # Can maybe clamp the output instead
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(-1, self.CHANNELS5 * 2 * 2)
        x = self.encoder_linear(x)
        mu = self.encoder_mu(x)
        log_sigma = self.encoder_log_sigma(x)

        return mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_linear(z)
        x = x.view(-1, self.CHANNELS5, 2, 2)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        x = transforms.Normalize(self.input_mean, self.input_std)(x)
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_sigma, z

    def convEncodeShape(self, x):
        x = self.encoder_conv(x)
        return x.shape

    # https://github.com/pytorch/examples/blob/a74badde33f924c2ce5391141b86c40483150d5a/vae/main.py#L73
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, x, recon_x, mu, logvar):
        #  criterion = nn.MSELoss(reduction='sum')
        #  MSE = criterion(recon_x, x)
        #  MSE /= x.size(0)
        size = x.size(1) * x.size(2) * x.size(3)

        BCE = F.binary_cross_entropy_with_logits(
            recon_x.view(-1, size), x.view(-1, size), reduction="sum"
        )
        BCE /= x.size(0)  # Normalize for batch size

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= x.size(0)  # Normalize for batch size
        #  print(f"KLD: {KLD}")

        loss = (BCE + KLD) / x.shape[1] # Normalize for channel count
        return loss

        #  return MSE + KLD

    def training_step(self, batch, batch_idx):
        outputs, mu, log_sigma, z = self.forward(batch)
        loss = self.loss_function(batch, outputs, mu, log_sigma)
        self.log('Loss/train', loss, sync_dist=True)

        # Only log images from the first GPU
        if self.global_step % 500 == 0 and self.global_rank == 0:
            self.logger.experiment.add_images(
                'Train/inputs', flattenChannels(batch), self.global_step
            )
            self.logger.experiment.add_images(
                'Train/outputs', flattenChannels(nn.Sigmoid()(outputs)),
                self.global_step
            )

        return loss


    def validation_step(self, batch, batch_idx):
        outputs, mu, log_sigma, z = self.forward(batch)
        loss = self.loss_function(batch, outputs, mu, log_sigma)
        self.log('Loss/val', loss, sync_dist=True)
        
        # Only log images from the first GPU
        if batch_idx == 0 and self.global_rank == 0:
            self.logger.experiment.add_images(
                'Val/inputs', flattenChannels(batch), self.global_step
            )
            self.logger.experiment.add_images(
                'Val/outputs', flattenChannels(nn.Sigmoid()(outputs)),
                self.global_step
            )

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

def main():
    train_loader, val_loader = loaders.makeLoaders(
            train_batch=32,
            test_batch=32,
            channels=args.channels)
    net = Autoencoder(args.hidden, args.channels)

    logger = pl.loggers.TensorBoardLogger('/data/ptl', name=args.comment)

    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=logger,
        #  callbacks = [EarlyStopping(monitor='Loss/val', min_delta=1)],
    )
    trainer.fit(net, train_loader, val_loader)


if __name__ == "__main__":
    main()
