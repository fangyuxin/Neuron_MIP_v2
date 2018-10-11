# coding:utf8
import warnings
import torch
from utils import *


class DefaultConfig(object):
    env = 'default'  # visdom 环境
    model = 'Unet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    dataset_root = './dataset'
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

    num_class = {'in': 1, 'out': 2}

    batch_size = 2  # batch size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 1  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    num_epochs = 30
    lr = 0.1  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-6  # 损失函数
    momentum = 0.9

    loss_name = None
    loss_param = {'weight': [1.0, 5.0]}


    optim_name = 'SGD'
    optim_param = {'SGD':
                      {'lr': lr,
                       'momentum': momentum,
                       'weight_decay': weight_decay
                       },
                   'Adam':
                      {'lr': lr,
                       'weight_decay': weight_decay
                       },
                   'RMSprop':
                      {'lr': lr,
                       'momentum': momentum,
                       'weight_decay': weight_decay
                       },
                   }


    shdlr_name = 'StepLR'
    shdlr_param = {
        'StepLR': {
            'step_size': 4,
            'gamma': 0.1
        },
    }

    metrics = 'IoU'

    softmax = nn.Softmax2d()


    def parse(self, kwargs):

        for k, v in kwargs.items():

            if not hasattr(self, k):
                warnings.warn("Warning: cfg has not attribut {}".format(k))

            setattr(self, k, v)


cfg = DefaultConfig()

