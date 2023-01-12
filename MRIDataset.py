import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import pandas as pd

class MRIDataset(Dataset):
    def __init__(self, annotation, image_dir, mask_dir=None, transform=None):
        self.annote = annotation
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        if mask_dir:
            self.target_transform = transform


    def __len__(self):
        return len(self.annote)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.annote.iloc[idx, 1])
        image = read_image(img_path)
        if self.mask_dir:
            mask_path = os.path.join(self.mask_dir, self.annote.iloc[idx, 2])
            mask = read_image(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask