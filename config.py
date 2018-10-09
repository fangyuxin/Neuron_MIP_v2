# coding:utf8
import warnings
import torch
from utils import *


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    model = 'unet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    dataset_root = '.dataset/'
    load_model_path = 'checkpoints/model.pth'  # 加载预训练的模型的路径，为None代表不加载

    # 数据增强
    train_aug_dict = {
        'Resize': 256,
        'RandomHorizontalFlip': None
    }
    other_aug_dict = {
        'Resize': 256
    }
    aug_dict = {
        'train': train_aug_dict,
        'other': other_aug_dict
    }

    in_channel = 1  # 输入通道数
    out_channel = 2  # 输出通道数

    batch_size = 128  # batch size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.1  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4  # 损失函数

    loss_name = None
    loss_param = None

    optim_name = None




    def parse(self, kwargs):

        for k, v in kwargs.items():

            if not hasattr(self, k):
                warnings.warn("Warning: cfg has not attribut {}".format(k))

            setattr(self, k, v)


cfg = DefaultConfig()




# def parse(self, kwargs):
#     '''
#     根据字典kwargs 更新 config参数
#     '''
#     for k, v in kwargs.iteritems():
#         if not hasattr(self, k):
#             warnings.warn("Warning: opt has not attribut %s" % k)
#         setattr(self, k, v)
#
#     print('user config:')
#     for k, v in self.__class__.__dict__.iteritems():
#         if not k.startswith('__'):
#             print(k, getattr(self, k))
#
#
# DefaultConfig.parse = parse
# opt = DefaultConfig()
# # opt.parse = parse
