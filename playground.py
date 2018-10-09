# from data import *
# import matplotlib.pyplot as plt
# import torch
#
#
#
# aug_dict = {
#     'train': {
#         'Resize': 256,
#         'RandomHorizontalFlip': random.randint(0,1),
#         'Grayscale': None
#     },
#
#     'other': {
#         'Resize': 256
#     }
# }
#
# for phase in ['train', 'val', 'test']:
#
#     neuron_dataset = NeuronDataset(phase=phase, aug_dict=aug_dict)
#
#     for i, (image, label) in enumerate(neuron_dataset):
#
#         plt.figure()
#
#         plt.subplot(1, 2, 1)
#         # print(image.numpy().shape)
#         plt.imshow(image.numpy()[0], cmap='gray')
#
#         plt.subplot(1, 2, 2)
#         plt.imshow(label.numpy()[0], cmap='gray')
#
#         plt.show()
#
#         if i == 2:
#             break
#
#
#
# from config import DefaultConfig
#
# cfg = DefaultConfig()
#
# print(cfg.lr)
#
# cfg.parse({'lr': 0.001})
#
# print(cfg.lr)


def add(a=1, b=2, c=3):

    return a + b + c

dict = {'a': 3, 'b': 2}
print(add(**dict))

