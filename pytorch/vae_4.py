#!/usr/bin/env python3
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--comment", default="autoencoder",
        help="Comment to append to folder name")
args = parser.parse_args()

# In[128]:


import torch
import torchvision
import torchvision.transforms as transforms


# In[129]:


import PIL
import os
import pathlib

# https://pytorch.org/docs/stable/notes/randomness.html 
torch.manual_seed(42)
torch.set_deterministic(True)

class FlatImageFolder(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        self._path = pathlib.Path(path)
        self._transform = transform
        self._image_paths = list(self._path.iterdir())
        
    def __len__(self):
        return len(self._image_paths)
    
    def __getitem__(self, idx):
        with open(self._image_paths[idx], 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert("L")
            
            return self._transform(img)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Pad(7),
])

train_dataset = FlatImageFolder(
        "/data/datasets/fonts/rendered/alphabet_upper_split_05/train/Q", transform)
print(f"Train dataset has {len(train_dataset)} examples")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=10)
train_iter = iter(train_loader)
train_images = train_iter.next()

test_dataset = FlatImageFolder(
        "/data/datasets/fonts/rendered/alphabet_upper_split_05/test/Q", transform)
print(f"Test dataset has {len(test_dataset)} examples")

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=10)
test_iter = iter(test_loader)

import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#  imshow(torchvision.utils.make_grid(train_images[:16]))
print(train_images.shape)


# In[130]:


import torch.nn as nn
import torch.nn.functional as F
    
# https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
# https://arxiv.org/pdf/1808.00362.pdf
class Autoencoder(nn.Module):
    # Pytorch uses sqrt(5) for the value of a in the kaiming uniform
    # initialization, so that's what I'm going to do. It seems to provide
    # significantly faster convergence than using a leak of .2.
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L112
    # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L87
    RELU_LEAK = 5 ** 0.5
    def __init__(self, hidden):
        super(Autoencoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
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

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(self.RELU_LEAK),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),

            nn.LeakyReLU(self.RELU_LEAK),
            nn.Conv2d(8, 512, 1),

            nn.LeakyReLU(self.RELU_LEAK),
            nn.Conv2d(512, 512, 1),

            nn.LeakyReLU(self.RELU_LEAK),
            nn.Conv2d(512, 1, 1),

            nn.Sigmoid() # Can maybe clamp the output instead
        )
        
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
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        reconstructed = self.decode(z)
        return reconstructed, mu, log_sigma, z


net = Autoencoder(16)
n, mu, log_sigma, z = net(train_images)
print("Batch output shape:", n.shape)
print()

for i, p in enumerate(net.parameters()):
    print("Parameters", i, p.size())
print("Trainable parameters:", sum([p.numel() for p in net.parameters()]))

#  import sys
#  sys.exit()

# In[131]:


# https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  net = nn.DataParallel(net)

net.to(device)


# In[132]:


import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

writer = SummaryWriter(comment=f"_{args.comment}")

optimizer = optim.Adam(net.parameters(), lr=.001)

criterion = nn.MSELoss(reduction='sum')
# https://github.com/pytorch/examples/blob/a74badde33f924c2ce5391141b86c40483150d5a/vae/main.py#L73 
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x, recon_x, mu, logvar):
    #  MSE = criterion(recon_x, x) 
    #  MSE /= x.size(0)
    size = x.size(1) * x.size(2) * x.size(3)

    BCE = F.binary_cross_entropy(recon_x.view(-1, size), x.view(-1, size), reduction='sum')
    BCE /= x.size(0) # Normalize for batch size

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= x.size(0) # Normalize for batch size

    return BCE + KLD
    #  return MSE + KLD

import time
start = time.time()

global_step = 0
for epoch in range(3):
    
    for train_minibatch, train_inputs in enumerate(train_loader):
        global_step += 1
        
        train_inputs = train_inputs.to(device)
        
        optimizer.zero_grad()
        train_outputs, train_mu, train_log_sigma, train_z = net(train_inputs)
        # Loss should be normalized to per-data-point
        train_loss = loss_function(train_inputs, train_outputs, train_mu, train_log_sigma)
        train_loss.backward()
        optimizer.step()
        
        writer.add_scalar("Loss/train", train_loss.item(), global_step)
        
        with torch.no_grad():
            if global_step % 200 == 0:
                print("[Step {:5d}] train loss: {:0.5f}, ".format(global_step, train_loss.item()), end = "")
                
                writer.add_images("Train/inputs", train_inputs[:128], global_step)
                writer.add_images("Train/outputs", train_outputs[:128], global_step)
            
                # Now run through the full test dataset
                test_loss = 0
                test_total = 0

                test_display_inputs = None
                test_display_outputs = None

                for test_minibatch, test_inputs in enumerate(test_loader):
                    test_inputs = test_inputs.to(device)
    
                    test_outputs, test_mu, test_log_sigma, test_z = net(test_inputs)
        
                    minibatch_test_loss = loss_function(test_inputs, test_outputs, test_mu, test_log_sigma)
                    test_loss += minibatch_test_loss * test_inputs.size(0)
                    test_total += test_inputs.size(0)

                    if (test_display_inputs is None or 
                            test_inputs.size(0) >= test_display_inputs.size(0)):
                        test_display_inputs = test_inputs

                    if (test_display_outputs is None or 
                            test_outputs.size(0) >= test_display_outputs.size(0)):
                        test_display_outputs = test_outputs

                test_loss /= test_total # To calculate avg per-element loss

                print("test loss: {:0.5f}".format(test_loss))
                
                writer.add_scalar("Loss/test", test_loss.item(), global_step)
                writer.add_images("Test/inputs", test_display_inputs[:128], global_step)
                writer.add_images("Test/outputs", test_display_outputs[:128], global_step)
                writer.add_embedding(test_z, label_img = test_inputs, 
                        global_step=global_step, tag="Test/embedding")

                writer.add_embedding(train_z, label_img = train_inputs, 
                        global_step=global_step, tag="Train/embedding")

writer.close()

end = time.time()
            
print(f"Finished training in {(end - start):0.2f} seconds!")

