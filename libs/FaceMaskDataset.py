from typing import Tuple
import os
import torch
import pandas as pd
from torch._C import dtype
from torchvision.io import read_image
from torch.utils.data import Dataset


class FaceMaskDataset(Dataset):
    def __init__(self, annotations, path, transform=None, target_transform=None) -> None:
        super().__init__()
        self.annotations = pd.read_csv(annotations)
        self.path = path
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        img = read_image(os.path.join(self.path, self.annotations.iloc[index, 1]))
        label = self.annotations.iloc[index, 0]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img.float(), label
