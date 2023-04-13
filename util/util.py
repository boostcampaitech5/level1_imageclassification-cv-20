import os
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_age_label(train_data, base_path):
    """
    Args:
        train_data (pandas.DataFrame): train.csv
        base_path (string): text with train image path

    Return:
        tuple of images path and images label
    """
    images = []
    labels = []

    for p in train_data["path"]:
        _, gender, _, age = p.split("_")

        path = os.path.join(base_path, p)

        num = 0
        if int(age) < 20:  # 0-19
            pass
        elif (int(age) >= 20) & (int(age) < 30):  # 20-29
            num = 1
        elif (int(age) >= 30) & (int(age) < 50):  # 30-49
            num = 2
        elif (int(age) >= 50) & (int(age) < 60):  # 50-59
            num = 3
        elif int(age) >= 60:
            num = 4

        for s in glob.glob(path + "/*"):
            name = s.split(os.sep)[-1]
            if name.find("incorrect") == 0:
                images.append(os.path.join(path, name))
                labels.append(num)

            elif name.find("mask") == 0:
                images.append(os.path.join(path, name))
                labels.append(num)

            elif name.find("normal") == 0:
                images.append(os.path.join(path, name))
                labels.append(num)

    return images, labels


def get_train_transform():
    """
    Return:
        transforms object
    """
    train_transform = A.Compose(
        [
            A.CenterCrop(320, 320, p=1),
            A.HorizontalFlip(p=0.5),
            A.GaussianBlur((3, 7), 3, p=0.5),
            A.ColorJitter(0.1, 0.2, 0.1, 0.1, p=0.5),
            A.Sharpen((0.4, 0.7), p=1),
            A.Normalize(p=1),
            ToTensorV2(),
        ]
    )

    return train_transform
