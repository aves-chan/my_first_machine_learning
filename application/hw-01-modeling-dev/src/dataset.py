import os
from typing import Optional

import albumentations as albu
import cv2
import numpy as np
from torch.utils.data import Dataset


class PlanetDataset(Dataset):
    def __init__(
        self,
        image_names: np.array,
        image_labels: np.array,
        image_folder: str,
        transforms: Optional[albu.BaseCompose] = None,
    ):
        self.image_names = image_names
        self.image_labels = image_labels
        self.image_folder = image_folder
        self.transforms = transforms

    def __getitem__(self, idx: int):
        image_path = os.path.join(self.image_folder, f"{self.image_names[idx]}.jpg")
        labels = self.image_labels[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data = {"image": image, "labels": labels}

        if self.transforms:
            data = self.transforms(**data)

        return data["image"], data["labels"]

    def __len__(self) -> int:
        return len(self.image_names)
