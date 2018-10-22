# # from data import *
# # # import matplotlib.pyplot as plt
# # # import torch
# # #
# # #
# # #
# # # aug_dict = {
# # #     'train': {
# # #         'Resize': 256,
# # #         'RandomHorizontalFlip': random.randint(0,1),
# # #         'Grayscale': None
# # #     },
# # #
# # #     'other': {
# # #         'Resize': 256
# # #     }
# # # }
# # #
# # # for phase in ['train', 'val', 'test']:
# # #
# # #     neuron_dataset = NeuronDataset(phase=phase, aug_dict=aug_dict)
# # #
# # #     for i, (image, label) in enumerate(neuron_dataset):
# # #
# # #         plt.figure()
# # #
# # #         plt.subplot(1, 2, 1)
# # #         # print(image.numpy().shape)
# # #         plt.imshow(image.numpy()[0], cmap='gray')
# # #
# # #         plt.subplot(1, 2, 2)
# # #         plt.imshow(label.numpy()[0], cmap='gray')
# # #
# # #         plt.show()
# # #
# # #         if i == 2:
# # #             break
# # #
# # #
# # #
# # # from config import DefaultConfig
# # #
# # # cfg = DefaultConfig()
# # #
# # # print(cfg.lr)
# # #
# # # cfg.parse({'lr': 0.001})
# # #
# # # print(cfg.lr)
# # import os
# # import yaml
# #
# # cfg = yaml.load(open('cfg.yml').read())
# # print(cfg['training'])
# #
# # for epoch in range(opt.max_epoch):
# #
# #     loss_meter.reset()
# #     confusion_matrix.reset()
# #
# #     for ii, (data, label) in tqdm(enumerate(train_dataloader), total=len(train_data)):
# #
# #         # train model
# #         input = Variable(data)
# #         target = Variable(label)
# #         if opt.use_gpu:
# #             input = input.cuda()
# #             target = target.cuda()
# #
# #         optimizer.zero_grad()
# #         score = model(input)
# #         loss = criterion(score, target)
# #         loss.backward()
# #         optimizer.step()
# #
# #         # meters update and visualize
# #         loss_meter.add(loss.data[0])
# #         confusion_matrix.add(score.data, target.data)
# #
# #         if ii % opt.print_freq == opt.print_freq - 1:
# #             vis.plot('loss', loss_meter.value()[0])
# #
# #             # 进入debug模式
# #             if os.path.exists(opt.debug_file):
# #                 import ipdb;
# #
# #                 ipdb.set_trace()
# #
# #     model.save()
# #
# #     # validate and visualize
# #     val_cm, val_accuracy = val(model, val_dataloader)
# #
# #     vis.plot('val_accuracy', val_accuracy)
# #     vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
# #         epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
# #         lr=lr))
# #
# #     # update learning rate
# #     if loss_meter.value()[0] > previous_loss:
# #         lr = lr * opt.lr_decay
# #         # 第二种降低学习率的方法:不会有moment等信息的丢失
# #         for param_group in optimizer.param_groups:
# #             param_group['lr'] = lr
# #
# #     previous_loss = loss_meter.value()[0]

# import torch
# from models import *
# model = Unet()
# model.load("./checkpoints/<class 'models.unet.Unet'>_best_IoU: 0.8070  best_Dice: 0.8815  ")
# for param in model.state_dict():
#     print(param)
# # t = torch.ones(3, 2)
# # print(t.view(-1, *t.size()).size())

#
# a = 1
#
# def f(b):
#     b = 2
#
# print(a)
# f(a)
# print(a)
#
# #
# from torchvision.models import densenet

#
# print('\'')
#
# import csv
#
# def dict2csv_r(dict, file):
#     with open(file, 'a+') as f:
#         w=csv.writer(f)
#         # write each key/value pair on a separate row
#         # w.writerow(dict.keys())
#         w.writerow(list)
#
#
# def dict2csv_c(dict, file):
#     with open(file, 'w+') as f:
#         w=csv.writer(f)
#         # write each key/value pair on a separate row
#         w.writerows(dict.items())
#
#
# dict = {'image_1': {'IoU': 90, 'Dice': 80},
#         'image_2': {'IoU': 100, 'Dice': 60}}
#
# dict_ = {'a': 1, 'b': 2}
#
# list = ['str', '1', 2]
#
# dict2csv_r(list, './rec.csv')

# dict_ = {'a': 1, 'b': 2}
#
# list = [x for x in dict_.values()]
#
# list.insert(0, 1000)
# print(list)


# print( 1 + 1e-20)
# -*- coding: utf-8 -*-

#
# class Celsius():
#     def __init__(self, temperature=0):
#         self.temperature = temperature
#
#     def get_temperature(self):
#         print('Getting value')
#         return self._temperature
#
#     def set_temperature(self, value):
#         if value < -273:
#             raise ValueError('Wrong')
#         print('Setting value')
#         self._temperature = value
#
#     temperature = property(get_temperature, set_temperature)
#
# c = Celsius()
# print(c.temperature)

def f1(msg):

    def printer():
        print(msg)

    return printer()


def f2(msg):
    def printer():
        print(msg)

    return printer

f1('Hello')
f = f2('Hello')
f()
del f2
# f_ = f2('Hello')
f()