from data import *
import matplotlib.pyplot as plt
import torch

aug_dict = {
    'train': {
        'Resize': 256,
        'RandomHorizontalFlip': None
    },

    'other': {
        'Resize': 256,

    }
}
for phase in ['train', 'val', 'test']:

    neuron_dataset = NeuronDataset(phase=phase)
    for i, (image, label) in enumerate(neuron_dataset):

        plt.figure()

        plt.subplot(1, 2, 1)
        # print(image.numpy().shape)
        plt.imshow(image.numpy()[0])

        plt.subplot(1, 2, 2)
        plt.imshow(label.numpy()[0])

        plt.show()

        if i == 2:
            break

