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
    def __init__(self, hidden):
        super(Autoencoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 4, stride=1, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
            antialiased_cnns.BlurPool(16, stride=2),
        
            nn.Conv2d(16, 32, 4, stride=1, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
            antialiased_cnns.BlurPool(32, stride=2),
            
            nn.Conv2d(32, 64, 4, stride=1, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
            antialiased_cnns.BlurPool(64, stride=2),
            
            nn.Conv2d(64, 128, 4, stride=1, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
            antialiased_cnns.BlurPool(128, stride=2),
            
            nn.Conv2d(128, 256, 4, stride=1, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
            antialiased_cnns.BlurPool(256, stride=2),
        )
        
        self.encoder_linear = nn.Sequential(
            nn.Linear(256*2*2, 256),
            nn.LeakyReLU(self.RELU_LEAK),
        )
        
        self.encoder_mu = nn.Sequential(
            nn.Linear(256, hidden), # No activation, let it go wherever
        )

        self.encoder_log_sigma = nn.Sequential(
            nn.Linear(256, hidden), # Will be exponentiated later
        )

        self.decoder_linear = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.LeakyReLU(self.RELU_LEAK),
            nn.Linear(256, 256*2*2),
            nn.LeakyReLU(self.RELU_LEAK),
        )

        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(self.RELU_LEAK),
            ),
            
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(self.RELU_LEAK),
            ),
            
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(self.RELU_LEAK),
            ),
            
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(self.RELU_LEAK),
            ),
            
            nn.Sequential(
                nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
                nn.LeakyReLU(self.RELU_LEAK),
            ),
        ])
        
        self.decoder_fc = nn.Sequential(
            # Begin "fully connected layers"
            nn.Conv2d(248, 512, 1),
            nn.LeakyReLU(self.RELU_LEAK),
        
            nn.Conv2d(512, 512, 1),
            nn.LeakyReLU(self.RELU_LEAK),
        
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid() # Can maybe clamp the output instead
        )

        #  self.decoder_conv = nn.Sequential(
        #      nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
        #      nn.LeakyReLU(self.RELU_LEAK),
        #  
        #      nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
        #      nn.LeakyReLU(self.RELU_LEAK),
        #  
        #      nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        #      nn.LeakyReLU(self.RELU_LEAK),
        #  
        #      nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        #      nn.LeakyReLU(self.RELU_LEAK),
        #  
        #      nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
        #      nn.LeakyReLU(self.RELU_LEAK),
        #     
        #      # Begin "fully connected layers"
        #      nn.Conv2d(8, 512, 1),
        #      nn.LeakyReLU(self.RELU_LEAK),
        #  
        #      nn.Conv2d(512, 512, 1),
        #      nn.LeakyReLU(self.RELU_LEAK),
        #  
        #      nn.Conv2d(512, 1, 1),
        #      nn.Sigmoid() # Can maybe clamp the output instead
        #  )
        
    def encode(self, x):
        x = self.encoder_conv(x)
        x = x.view(-1, 256*2*2)
        x = self.encoder_linear(x)
        mu = self.encoder_mu(x)
        log_sigma = self.encoder_log_sigma(x)
        
        return mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5*log_sigma)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        x = self.decoder_linear(z)
        x = x.view(-1, 256, 2, 2)
        #  x = self.decoder_conv(x)

        upsampled = []
        upsampler = nn.Upsample(size=(64, 64), mode="bilinear",
                align_corners=False)
        for layer in self.decoder_layers:
            x = layer(x)
            upsampled.append(upsampler(x))

        upsampled_cat = torch.cat(upsampled, dim=1)
        
        x = self.decoder_fc(upsampled_cat)
        #  x = self.decoder_fc(x)

        return x

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_sigma, z

    def convEncodeShape(self, x):
        x = self.encoder_conv(x)
        return x.shape
