#!/usr/bin/env python3
# coding: utf-8

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--comment",
    default="autoencoder",
    help="Comment to append to folder name",
)
parser.add_argument(
    "-n",
    "--channels",
    default=1,
    type=int,
    help="Number of channels",
)
parser.add_argument(
    "-e",
    "--epochs",
    default=3,
    type=int,
    help="Number of training epochs",
)
args = parser.parse_args()

import time
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model_simple import Autoencoder
from loaders import makeLoaders

def flattenChannels(stacked):
    N, C, H, W = stacked.shape
    channels = [stacked[:, c, :, :].view(N, 1, H, W) for c in range(C)]
    return torch.cat(channels, 2)

def makeModel(train_loader):
    # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = Autoencoder(16, args.channels)
    net.to(device)

    train_images = next(iter(train_loader))
    train_images = train_images.to(device)
    print("Image input shape:        ", train_images.shape)
    print("Encoder conv output shape:", net.convEncodeShape(train_images))
    n, mu, log_sigma, z = net(train_images)
    print("Decoder output shape:     ", n.shape)
    print()

    for i, p in enumerate(net.parameters()):
        print("Parameters", i, p.size())
    print("Trainable parameters:", sum([p.numel() for p in net.parameters()]))
    print()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
        net.to(device)

    return net, device


# https://github.com/pytorch/examples/blob/a74badde33f924c2ce5391141b86c40483150d5a/vae/main.py#L73
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(x, recon_x, mu, logvar):
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


def train():
    # setup
    # https://pytorch.org/docs/stable/notes/randomness.html
    torch.set_deterministic(True)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    # model stuff here
    train_loader, test_loader = makeLoaders(channels=args.channels)
    net, device = makeModel(train_loader)

    #  return
    
    optimizer = optim.AdamW(net.parameters(), lr=0.0001)
    writer = SummaryWriter(comment=f"_{args.comment}")
    scaler = torch.cuda.amp.GradScaler()

    start = time.time()

    global_step = 0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        for train_minibatch, train_inputs in enumerate(train_loader):
            global_step += 1

            # Training stuff
            train_start = time.time()
            train_inputs = train_inputs.to(device)

            torch.set_grad_enabled(True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():

            #  with torch.autograd.detect_anomaly():
                train_outputs, train_mu, train_log_sigma, train_z = net(
                    train_inputs
                )
                # Loss should be normalized to per-data-point
                train_loss = loss_function(
                    train_inputs, train_outputs, train_mu, train_log_sigma
                )

            scaler.scale(train_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #  train_loss.backward()
            #  optimizer.step()

            train_end = time.time()
            train_time = train_end - train_start

            writer.add_scalar("Loss/train", train_loss.item(), global_step)
            writer.add_scalar("Train/time", train_time, global_step)
            torch.set_grad_enabled(False)

            if global_step % 200 == 0:
                print(
                    "[Step {:5d}] train loss: {:0.5f}, ".format(
                        global_step, train_loss.item()
                    ),
                    end="",
                )

                writer.add_images(
                    "Train/inputs", flattenChannels(train_inputs), global_step
                )
                writer.add_images(
                    "Train/outputs", flattenChannels(nn.Sigmoid()(train_outputs)), global_step
                )

            # Only test every so often
            if global_step % 200 != 0:
                continue

            # Testing stuff
            test_start = time.time()
            test_loss = 0
            test_total = 0

            for test_minibatch, test_inputs in enumerate(test_loader):
                test_inputs = test_inputs.to(device)

                with torch.cuda.amp.autocast():
                    test_outputs, test_mu, test_log_sigma, test_z = net(
                        test_inputs
                    )

                    minibatch_test_loss = loss_function(
                        test_inputs, test_outputs, test_mu, test_log_sigma
                    )

                test_loss += minibatch_test_loss * test_inputs.size(0)
                test_total += test_inputs.size(0)
            
            test_loss /= test_total  # To calculate avg per-element loss
            test_end = time.time()

            print("test loss: {:0.5f}".format(test_loss))

            writer.add_scalar(
                "Loss/test", test_loss.item(), global_step
            )
            writer.add_images(
                "Test/inputs", flattenChannels(test_inputs), global_step,
            )
            writer.add_images(
                "Test/outputs", flattenChannels(nn.Sigmoid()(test_outputs)), global_step,
            )
            writer.add_scalar(
                "Test/time", test_end - test_start, global_step
            )

    writer.close()

    end = time.time()

    print(f"Finished training in {(end - start):0.2f} seconds!")


def main():
    train()


if __name__ == "__main__":
    main()
