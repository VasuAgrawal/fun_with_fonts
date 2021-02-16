import pathlib
import PIL
from quantization import *

import torch
import torchvision
import torchvision.transforms as transforms


class FlatImageFolder(torch.utils.data.Dataset):
    def __init__(self, path, transform, channels, buckets):
        self._path = pathlib.Path(path)
        self._transform = transform
        self._image_paths = list(self._path.glob("**/*.pgm"))
        self._channels = channels
        self._buckets = buckets

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        with open(self._image_paths[idx], "rb") as f:
            img = PIL.Image.open(f)
            img = img.convert("L")
            height = img.size[1]
            letters = [
                img.crop((i * height, 0, (i + 1) * height, height))
                for i in range(img.size[0] // height)
            ]

            tensored = self._transform(letters[: self._channels])
            quantized = quantizeBatch(torch.unsqueeze(tensored, 0), self._buckets)
            labeled = makeQuantizedLabels(quantized, self._buckets)
            labeled = [torch.squeeze(l, 0) for l in labeled]
            return tensored, labeled


def makeLoaders(train_batch=64, test_batch=128, channels=1, size=64,
        buckets=2):
    transform = transforms.Compose(
        [
            transforms.Lambda(
                lambda letters: torch.cat(
                    [transforms.ToTensor()(l) for l in letters], 0
                )
            ),
        ]
    )

    train_dataset = FlatImageFolder(
        f"/data/datasets/fonts/rendered/ocr_line_split_05_val/{size}/train",
        transform,
        channels,
        buckets
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
        f"/data/datasets/fonts/rendered/ocr_line_split_05_val/{size}/validation",
        transform,
        channels,
        buckets
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )

    meanstd = torch.load(
        "/data/datasets/fonts/rendered/ocr_line_split_05_val/64/meanstd.pt"
    )

    return train_loader, test_loader, meanstd
