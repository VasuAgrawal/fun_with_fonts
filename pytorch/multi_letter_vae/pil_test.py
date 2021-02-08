#!/usr/bin/env python3

import pathlib
import PIL
import PIL.Image

import torch
import torchvision
import torchvision.transforms as transforms

folder = "/data/datasets/fonts/rendered/ocr_line_split_05_val/64/train"
folder = pathlib.Path(folder)
image_paths = list(folder.glob("**/*.pgm"))

images = []
for path in image_paths[:10]:
    with open(path, "rb") as f:
        img = PIL.Image.open(f)
        img = img.convert("L")

        height = img.size[1]
        letters = [
            img.crop((i * height, 0, (i + 1) * height, height)) for i in
            range(img.size[0] // height)
        ]

        transform = transforms.Compose([
            transforms.Lambda(
                lambda letters: torch.cat([
                    transforms.ToTensor()(l) for l in letters], 0)),
        ])

        tensor = transform(letters[:2])
        images.append(tensor)

stacked = torch.stack(images)

def flattenChannels(stacked):
    N, C, H, W = stacked.shape
    channels = [stacked[:, c, :, :].view(N, 1, H, W) for c in range(C)]
    return torch.cat(channels, 2)

flat = flattenChannels(stacked)
print(flat.shape)
