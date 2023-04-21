import os
import glob
from collections import OrderedDict
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_labels(train_data, base_path):
    """
    Args:
        train_data (pandas.DataFrame): train.csv
        base_path (string): text with train image path

    Return:
        tuple of images path and images label
    """
    images = []
    labels = []
    mask_label = []
    gender_label = []
    age_label = []

    for p in train_data["path"]:
        _, gender, _, age = p.split("_")

        path = os.path.join(base_path, p)
        num = 0

        age_num = 0
        if (int(age) >= 30) & (int(age) < 60):
            num += 1
            age_num = 1
        elif int(age) >= 60:
            num += 2
            age_num = 2

        gender_num = 0
        if gender == "female":
            num += 3
            gender_num = 1

        for s in glob.glob(path + "/*"):
            name = s.split(os.sep)[-1]
            if name.find("incorrect") == 0:
                images.append(os.path.join(path, name))
                labels.append(num + 6)
                mask_label.append(1)
                gender_label.append(gender_num)
                age_label.append(age_num)
            elif name.find("mask") == 0:
                images.append(os.path.join(path, name))
                labels.append(num)
                mask_label.append(0)
                gender_label.append(gender_num)
                age_label.append(age_num)
            elif name.find("normal") == 0:
                images.append(os.path.join(path, name))
                labels.append(num + 12)
                mask_label.append(2)
                gender_label.append(gender_num)
                age_label.append(age_num)

    return images, (mask_label, gender_label, age_label)


def get_train_transform():
    """
    Return:
        transforms object
    """
    train_transform = A.Compose(
        [
            A.CenterCrop(320, 320, p=1),
            A.HorizontalFlip(p=0.5),
            A.Cutout(num_holes=80, max_h_size=32, max_w_size=32, fill_value=0, p=1),
            A.Normalize(p=1),
            ToTensorV2(),
        ]
    )

    return train_transform


def assign_pretrained_weight(model, weight_path):
    """
    Args:
        model (torch.module): your model
        weight_path (string): weight file path

    Return:
        torch.module with assignment pretrained weight
    """
    weight = torch.load(weight_path)
    pretrained = [
        "conv1.weight",
        "bn1.running_mean",
        "bn1.running_var",
        "bn1.weight",
        "bn1.bias",
        "layer1.0.conv1.weight",
        "layer1.0.bn1.running_mean",
        "layer1.0.bn1.running_var",
        "layer1.0.bn1.weight",
        "layer1.0.bn1.bias",
        "layer1.0.conv2.weight",
        "layer1.0.bn2.running_mean",
        "layer1.0.bn2.running_var",
        "layer1.0.bn2.weight",
        "layer1.0.bn2.bias",
        "layer1.1.conv1.weight",
        "layer1.1.bn1.running_mean",
        "layer1.1.bn1.running_var",
        "layer1.1.bn1.weight",
        "layer1.1.bn1.bias",
        "layer1.1.conv2.weight",
        "layer1.1.bn2.running_mean",
        "layer1.1.bn2.running_var",
        "layer1.1.bn2.weight",
        "layer1.1.bn2.bias",
        "layer2.0.conv1.weight",
        "layer2.0.bn1.running_mean",
        "layer2.0.bn1.running_var",
        "layer2.0.bn1.weight",
        "layer2.0.bn1.bias",
        "layer2.0.conv2.weight",
        "layer2.0.bn2.running_mean",
        "layer2.0.bn2.running_var",
        "layer2.0.bn2.weight",
        "layer2.0.bn2.bias",
        "layer2.1.conv1.weight",
        "layer2.1.bn1.running_mean",
        "layer2.1.bn1.running_var",
        "layer2.1.bn1.weight",
        "layer2.1.bn1.bias",
        "layer2.1.conv2.weight",
        "layer2.1.bn2.running_mean",
        "layer2.1.bn2.running_var",
        "layer2.1.bn2.weight",
        "layer2.1.bn2.bias",
        "layer3.0.conv1.weight",
        "layer3.0.bn1.running_mean",
        "layer3.0.bn1.running_var",
        "layer3.0.bn1.weight",
        "layer3.0.bn1.bias",
        "layer3.0.conv2.weight",
        "layer3.0.bn2.running_mean",
        "layer3.0.bn2.running_var",
        "layer3.0.bn2.weight",
        "layer3.0.bn2.bias",
        "layer3.1.conv1.weight",
        "layer3.1.bn1.running_mean",
        "layer3.1.bn1.running_var",
        "layer3.1.bn1.weight",
        "layer3.1.bn1.bias",
        "layer3.1.conv2.weight",
        "layer3.1.bn2.running_mean",
        "layer3.1.bn2.running_var",
        "layer3.1.bn2.weight",
        "layer3.1.bn2.bias",
        "layer4.0.conv1.weight",
        "layer4.0.bn1.running_mean",
        "layer4.0.bn1.running_var",
        "layer4.0.bn1.weight",
        "layer4.0.bn1.bias",
        "layer4.0.conv2.weight",
        "layer4.0.bn2.running_mean",
        "layer4.0.bn2.running_var",
        "layer4.0.bn2.weight",
        "layer4.0.bn2.bias",
        "layer4.1.conv1.weight",
        "layer4.1.bn1.running_mean",
        "layer4.1.bn1.running_var",
        "layer4.1.bn1.weight",
        "layer4.1.bn1.bias",
        "layer4.1.conv2.weight",
        "layer4.1.bn2.running_mean",
        "layer4.1.bn2.running_var",
        "layer4.1.bn2.weight",
        "layer4.1.bn2.bias",
    ]

    mymodel = [
        "conv.weight",
        "bn.running_mean",
        "bn.running_var",
        "bn.weight",
        "bn.bias",
        "layer1.0.conv1.weight",
        "layer1.0.bn1.running_mean",
        "layer1.0.bn1.running_var",
        "layer1.0.bn1.weight",
        "layer1.0.bn1.bias",
        "layer1.0.conv2.weight",
        "layer1.0.bn2.running_mean",
        "layer1.0.bn2.running_var",
        "layer1.0.bn2.weight",
        "layer1.0.bn2.bias",
        "layer1.1.conv1.weight",
        "layer1.1.bn1.running_mean",
        "layer1.1.bn1.running_var",
        "layer1.1.bn1.weight",
        "layer1.1.bn1.bias",
        "layer1.1.conv2.weight",
        "layer1.1.bn2.running_mean",
        "layer1.1.bn2.running_var",
        "layer1.1.bn2.weight",
        "layer1.1.bn2.bias",
        "layer2.0.conv1.weight",
        "layer2.0.bn1.running_mean",
        "layer2.0.bn1.running_var",
        "layer2.0.bn1.weight",
        "layer2.0.bn1.bias",
        "layer2.0.conv2.weight",
        "layer2.0.bn2.running_mean",
        "layer2.0.bn2.running_var",
        "layer2.0.bn2.weight",
        "layer2.0.bn2.bias",
        "layer2.1.conv1.weight",
        "layer2.1.bn1.running_mean",
        "layer2.1.bn1.running_var",
        "layer2.1.bn1.weight",
        "layer2.1.bn1.bias",
        "layer2.1.conv2.weight",
        "layer2.1.bn2.running_mean",
        "layer2.1.bn2.running_var",
        "layer2.1.bn2.weight",
        "layer2.1.bn2.bias",
        "layer3.0.conv1.weight",
        "layer3.0.bn1.running_mean",
        "layer3.0.bn1.running_var",
        "layer3.0.bn1.weight",
        "layer3.0.bn1.bias",
        "layer3.0.conv2.weight",
        "layer3.0.bn2.running_mean",
        "layer3.0.bn2.running_var",
        "layer3.0.bn2.weight",
        "layer3.0.bn2.bias",
        "layer3.1.conv1.weight",
        "layer3.1.bn1.running_mean",
        "layer3.1.bn1.running_var",
        "layer3.1.bn1.weight",
        "layer3.1.bn1.bias",
        "layer3.1.conv2.weight",
        "layer3.1.bn2.running_mean",
        "layer3.1.bn2.running_var",
        "layer3.1.bn2.weight",
        "layer3.1.bn2.bias",
        "layer4.0.conv1.weight",
        "layer4.0.bn1.running_mean",
        "layer4.0.bn1.running_var",
        "layer4.0.bn1.weight",
        "layer4.0.bn1.bias",
        "layer4.0.conv2.weight",
        "layer4.0.bn2.running_mean",
        "layer4.0.bn2.running_var",
        "layer4.0.bn2.weight",
        "layer4.0.bn2.bias",
        "layer4.1.conv1.weight",
        "layer4.1.bn1.running_mean",
        "layer4.1.bn1.running_var",
        "layer4.1.bn1.weight",
        "layer4.1.bn1.bias",
        "layer4.1.conv2.weight",
        "layer4.1.bn2.running_mean",
        "layer4.1.bn2.running_var",
        "layer4.1.bn2.weight",
        "layer4.1.bn2.bias",
    ]

    ordered_dic = OrderedDict()
    for i, j in zip(pretrained, mymodel):
        ordered_dic[j] = weight.get(i)

    model.load_state_dict(ordered_dic, strict=False)

    return model
