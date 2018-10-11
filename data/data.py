import os
import random
import numpy as np
from PIL import Image
from augmentations import *
from torch.utils.data import Dataset, DataLoader

class NeuronDataset(Dataset):

    def __init__(self, root='./dataset', phase='train', split_ratio=0.7, \
                 aug_dict={'train': None, 'other': None}):

        if phase == 'train':
            self.aug_phase = 'train'
        else:
            self.aug_phase = 'other'


        if phase == 'val':
            self.phase = 'train'
        else:
            self.phase = phase


        images_root = os.path.join(root, self.phase, 'image')
        images_list = [os.path.join(images_root, image_name) \
                        for image_name in os.listdir(images_root)]

        images_list = sorted(images_list, key=lambda x: \
                             int(x.split('.')[-2].split('/')[-1]))

        imgs_list_len = len(images_list)

        np.random.seed(100)
        np.random.shuffle(images_list)

        if phase == 'train':
            self.images_list = images_list[:int(imgs_list_len * split_ratio)]

        if phase == 'val':
            self.images_list = images_list[int(imgs_list_len * split_ratio):]

        if phase == 'test':
            self.images_list = images_list

        self.root = root
        self.transforms = get_composed_augmentations(aug_dict[self.aug_phase])

    def __getitem__(self, index):

        image_path = self.images_list[index]

        label_path = os.path.join(self.root, self.phase, 'label', \
                    '{}.{}'.format(image_path.split('.')[-2].split('/')[-1], \
                                   image_path.split('.')[-1]))

        image = Image.open(image_path)
        label = Image.open(label_path)

        seed = np.random.randint(2 ** 32)

        random.seed(seed)
        image = self.transforms(image)

        random.seed(seed)
        label = self.transforms(label)

        return image, label[0].long()

    def __len__(self):
        return len(self.images_list)







