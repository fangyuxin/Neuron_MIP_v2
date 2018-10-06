import torch
import random
import torchvision
import torchvision.transforms as T
import numpy as np

key2aug = {
    'Resize': T.Resize,
    'Grayscale': T.Grayscale,
    'RandomHorizontalFlip': T.RandomHorizontalFlip,
    'ToTensor': T.ToTensor
}

def get_composed_augmentations(aug_dict):

    augs = []

    if aug_dict is None:
        pass

    else:
        for aug_key, aug_param in aug_dict.items():
            augs.append((key2aug[aug_key](aug_param)))

    augs.append(T.ToTensor())

    return T.Compose(augs)