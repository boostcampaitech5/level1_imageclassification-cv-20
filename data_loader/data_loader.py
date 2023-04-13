import os
import pandas as pd
import numpy as np
from PIL import Image
from util.util import get_age_label
from torch.utils.data import Dataset, DataLoader


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


def get_trainloader(label_type, path, transform, batch_size, shuffle):
    """
    Args:
        label_type (string): Dataset type(Age, Gender, Mask)
        path (string): path with train.csv
        transform (Compose): transform object of torchvision or albumentations
        batch_size (int): train batch size
        shuffle (bool): shuffle at every epoch

    Return:
        torch.utils.data.dataloder: return dataloder object
    """
    if label_type == "age":
        dataset = Age_Dataset(path, transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
