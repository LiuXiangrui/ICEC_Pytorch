import torch
from torch.utils.data import Dataset
import numpy as np
import os


class MriDataset(Dataset):
    def __init__(self, root: str, split: str, plane_type: list = ("axial", "coronal", "sagittal"), transform=None) -> None:
        self.transform = transform

        self.data_list = []

        for plane in plane_type:
            for filepath in os.listdir(os.path.join(root, split, plane)):
                if os.path.splitext(filepath)[-1] != ".npy":
                    continue
                self.data_list.append(os.path.join(root, split, plane, filepath))

    def __getitem__(self, idx: int) -> torch.Tensor:
        data = np.load(self.data_list[idx])

        assert data.min() >= 0 and data.max() <= 255

        x = torch.from_numpy(data) * 1.

        return self.transform(x) if self.transform else x

    def __len__(self) -> int:
        return len(self.data_list)
