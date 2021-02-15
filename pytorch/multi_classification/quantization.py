#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import numpy as np
import cv2
import progressbar
from loaders import makeLoaders
from matplotlib import pyplot as plt

def quantizeBatch(images, values):
    # Images is a float tensor with [0, 1] values
    # Output is a float tensor with quantized [0, 1] floats
    return torch.round(images * (values - 1)) / (values - 1)


def makeQuantizedLabels(images, values):
    output = []
    for ch in range(images.size(1)):
        im = images[:, ch, :, :]
        scaled = im * (values - 1)
        label = torch.zeros(im.size(0), values, im.size(1), im.size(2), dtype=scaled.dtype)
        for v in range(values):
            label[:, v, :, :] = scaled == v
        output.append(label)

        summed = torch.sum(label, dim=1)
        assert torch.all(summed == torch.ones_like(summed))

    return output


def plotQuantizationLoss(loader):
    for i, data in enumerate(loader):
        x = np.array(range(2, 257))
        y = np.zeros_like(x, dtype=float)
        for i, values in progressbar.progressbar(enumerate(x)):
            quantized = quantizeBatch(data, values)
            loss = nn.MSELoss()(data, quantized)
            y[i] = loss

        for (values, loss) in zip(x, y):
            print(values, loss)

        plt.plot(x, y)
        plt.yscale('log')
        plt.show()

        break


def testQuantizedLabels(loader):
    buckets = 2
    for i, data in enumerate(loader):
        quantized = quantizeBatch(data, buckets)
        labels = makeQuantizedLabels(quantized, buckets)
    
        print("Original image:", data.shape)
        print(data)

        print("Quantized image", quantized.shape)
        print(quantized)

        for i, label in enumerate(labels):
            print(f"Label {i}", label.shape)
            print(label)

        return


def makeGrid(images):
    # Take a list of np images of equal size and turn it into a single grid.
    width = int(math.ceil(len(images) ** 0.5))
    height = int(math.ceil(len(images) / width))
    extra = (width * height) - len(images)

    blank = np.ones_like(images[0]) * 0.5
    h = blank.shape[0]
    w = blank.shape[1]
    images = images + [blank] * extra
    #  images = [im.reshape((-1, im.shape[-2], im.shape[-1])) for im in images]

    output = np.zeros((h * height, w * width), dtype=blank.dtype)
    count = 0
    for y in range(height):
        for x in range(width):
            output[y * h : (y+1) * h, x * w : (x + 1) * w] = images[count]
            count += 1

    return output


def plotQuantizedLabels(loader):

    max_buckets = 2
    for i, data in enumerate(loader):
        print(f"Showing image {i} channel 0")
        cv2.imshow("original", data[0, 0, :, :].numpy())
        #  print(data[0, 0, :, :])

        quantized = [quantizeBatch(data, q) for q in range(2, max_buckets + 1)]
        quantized_grid = makeGrid([im[0, 0, :, :].numpy() for im in quantized])
        print(f"Showing quantizations from 2 to {max_buckets + 1} buckets ({max_buckets - 1})")
        cv2.imshow("quantizations", quantized_grid)
        #  print(quantized[-1][0, 0, :, :])

        labels = makeQuantizedLabels(quantized[-1], max_buckets)
        labels_grid = makeGrid([labels[0][0, i, :, :].numpy()
            for i in range(max_buckets)])
        print(f"Showing labels for {max_buckets} buckets")
        cv2.imshow("labels", labels_grid)
        #  print(labels[-1])

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


train_loader, test_loader, meanstd = makeLoaders(1, 1, channels=1, size=128)
#  plotQuantizationLoss(train_loader)
#  testQuantizedLabels(train_loader)
plotQuantizedLabels(test_loader)

#  #  for i, data in progressbar.progressbar(train_loader):
#  for i, data in enumerate(train_loader):
#  
#      x = np.array(range(2, 257))
#      y = np.zeros_like(x, dtype=float)
#      #  for i, values in progressbar.progressbar(enumerate(x)):
#      for i, values in enumerate(x):
#          quantized = quantizeBatch(data, values)
#          labels = makeQuantizedLabels(quantized, values)
#          loss = nn.MSELoss()(data, quantized)
#          #  print(values, loss)
#          y[i] = loss
#  
#          break
#      break
#  
#      for (values, loss) in zip(x, y):
#          print(values, loss)
#  
#      plt.plot(x, y)
#      plt.yscale('log')
#      plt.show()
#  
#      break
