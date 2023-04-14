import os
import pandas as pd
import numpy as np
import torch
from PIL import Image
from util.cutmix import CutMixCollator
from util.util import get_age_label
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class Age_Dataset(Dataset):
    def __init__(self, path, transform):
        train_data = pd.read_csv(os.path.join(path, "train.csv"))
        base_path = os.path.join(path, "images")

        self.images, self.labels = get_age_label(train_data, base_path=base_path)
        self.transform = transform

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]))

        if self.transform != None:
            image = self.transform(image=img)["image"]

        label = self.labels[idx]

        return image, label

    def __len__(self):
        return len(self.images)


def get_trainloader(
    label_type, path, transform, batch_size, shuffle, weighted_sampler, collate
):
    """
    Args:
        label_type (string): Dataset type(Age, Gender, Mask)
        path (string): path with train.csv
        transform (Compose): transform object of torchvision or albumentations
        batch_size (int): train batch size
        shuffle (bool): shuffle at every epoch
        weighted_sampler (bool): use weighted sample

    Return:
        torch.utils.data.dataloder: return dataloder object
    """
    if label_type == "age":
        dataset = Age_Dataset(path, transform)

    if weighted_sampler:
        labels = np.array(dataset.labels)
        weight_arr = np.zeros_like(dataset.labels)

        _, counts = np.unique(labels, return_counts=True)
        for c in range(5):
            weight_arr = np.where(labels == c, 1 / counts[c], weight_arr)

        sampler = WeightedRandomSampler(weight_arr, 18900)
    else:
        sampler = None

    if collate:
        cutmix_collate = CutMixCollator(alpha=1.0)
    else:
        cutmix_collate = None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=cutmix_collate,
    )
