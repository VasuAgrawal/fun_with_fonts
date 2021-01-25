import pathlib
import PIL

import torch
import torchvision
import torchvision.transforms as transforms

class FlatImageFolder(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        self._path = pathlib.Path(path)
        self._transform = transform
        self._image_paths = list(self._path.glob("**/*.pgm"))

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        with open(self._image_paths[idx], "rb") as f:
            img = PIL.Image.open(f)
            img = img.convert("L")

            return self._transform(img)


def makeLoaders(train_batch = 64, test_batch = 128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(7),
        #  transforms.Normalize((.1357,), (.3297,)),
    ])

    train_dataset = FlatImageFolder(
        "/data/datasets/fonts/rendered/alphabet_upper_split_05/train/Q",
        transform,
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
        "/data/datasets/fonts/rendered/alphabet_upper_split_05/test/Q",
        transform,
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

