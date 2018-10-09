
import torch
from utils import *
from config import *
from models import *
import torch.nn as nn
from data import *
from utils import *


def train(**kwarg):

    '''
        1.初始化config, 更新相关参数.
        2.初始化visdom环境.
        3.初始化device.
        4.初始化model.
        5.初始化dataset, dataloader, augmentation.
    '''

    # 初始化config, 更新相关参数.
    cfg.parse(kwarg)

    # 初始化visdom环境.
    vis = Visualizer(cfg.env)

    # 初始化device.
    device = cfg.device

    # 初始化model.
    model = getattr(models, cfg.model)(cfg.in_channel, cfg.out_channel).to(device)
    model = nn.DataParallel(model)

    # 初始化dataset, dataloader, augmentation.
    train_data = NeuronDataset(cfg.dataset_root, phase='train', \
                               aug_dict=cfg.aug_dict)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, \
                              num_workers=cfg.num_workers)

    # 初始化criterion.
    criterion = get_criterion(cfg)

    # 初始化optimizer.
    optimizer = get_optimizer(cfg)(model.parameters(), \
                                   **cfg.optim_param[cfg.optim_name])




