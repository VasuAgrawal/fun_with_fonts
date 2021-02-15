import pathlib
import PIL

import torch
import torchvision
import torchvision.transforms as transforms

class FlatImageFolder(torch.utils.data.Dataset):
    def __init__(self, path, transform, channels):
        self._path = pathlib.Path(path)
        self._transform = transform
        self._image_paths = list(self._path.glob("**/*.pgm"))
        self._channels = channels

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        with open(self._image_paths[idx], "rb") as f:
            img = PIL.Image.open(f)
            img = img.convert("L")
            height = img.size[1]
            letters = [
                img.crop((i * height, 0, (i + 1) * height, height)) for i in
                range(img.size[0] // height)
                    ]

            return self._transform(letters[:self._channels])


def makeLoaders(train_batch = 64, test_batch = 128, channels = 1):
    transform = transforms.Compose([
        transforms.Lambda(
            lambda letters: torch.cat([
                transforms.ToTensor()(l) for l in letters], 0)),
        #  transforms.Normalize(
        #      [0.1222, 0.1541, 0.1076, 0.1491, 0.1232],
        #      [0.3176, 0.3502, 0.2987, 0.3454, 0.3199],
        #  )
    ])

    train_dataset = FlatImageFolder(
        "/data/datasets/fonts/rendered/ocr_line_split_05_val/64/train",
        transform,
        channels,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch,
        num_workers=8,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    test_dataset = FlatImageFolder(
        "/data/datasets/fonts/rendered/ocr_line_split_05_val/64/validation",
        transform,
        channels,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, test_loader

