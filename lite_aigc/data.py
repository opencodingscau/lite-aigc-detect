"""Dataset from jsonl manifests."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def build_transforms(size: int = 224, train: bool = True):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class JsonlImageDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, train: bool = False, size: int = 224, limit: int | None = None):
        self.rows = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        if limit is not None:
            self.rows = self.rows[:limit]
        self.tf = build_transforms(size=size, train=train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["path"]).convert("RGB")
        x = self.tf(img)
        y = int(r["label"])
        return x, y, r.get("source", "unk")
