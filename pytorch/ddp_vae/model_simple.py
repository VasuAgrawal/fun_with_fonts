import torch
import torch.nn as nn
import antialiased_cnns

# https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
# https://arxiv.org/pdf/1808.00362.pdf
class Autoencoder(nn.Module):
    # Pytorch uses sqrt(5) for the value of a in the kaiming uniform
    # initialization, so that's what I'm going to do. It seems to provide
    # significantly faster convergence than using a leak of .2.
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L112
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L87

    # Using blurpool to get antialiasing, not sure if it matters. Kernel size 4
    # seems to not make much of a difference, if at all.
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

    def __init__(self, hidden):
        super(Autoencoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, self.CHANNELS1, 4, stride=2, padding=1),
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

            nn.Conv2d(self.DECODER_FC3, 1, 1),
            nn.Sigmoid(),  # Can maybe clamp the output instead
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
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_sigma, z

    def convEncodeShape(self, x):
        x = self.encoder_conv(x)
        return x.shape
