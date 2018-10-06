import torch
import random
import torchvision
import torchvision.transforms as T
import numpy as np


class DataTransforms():

    def __init__(self, phase, seed=np.random.randint(2**32)):

        self.phase = phase
        random.seed(seed)

        self.data_transforms = {

            'train': T.Compose([
                T.Resize(256),
                T.RandomHorizontalFlip(),
                T.ToTensor()
            ]),

            'test': T.Compose([
                T.Resize(256),
                T.ToTensor()
            ])
        }


    def __call__(self, input):
        return self.data_transforms[self.phase](input)

def data_transforms(input, phase)




