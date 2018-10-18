
from config import *
import models
from data import *
from utils import *


def parse(**kwargs):
    cfg.parse(kwargs)

# 1.初始化config, 更新相关参数.
parse()

# 2.初始化visdom环境.
vis = Visualizer(cfg.env)

# 3.初始化device.
device = cfg.device

# 4.初始化model.
model = getattr(models, cfg.model) \
    (cfg.num_class['in'], cfg.num_class['out']).to(device)
model = nn.DataParallel(model)

# 5.初始化dataset, dataloader, augmentation.

if cfg.CV:

    dataset_list = []
    dataloader_list = []

    for i in range(cfg.k_fold_split):

        dataset_dict = {}
        dataloader_dict = {}

        for phase in ['train', 'val']:

            dataset = NeuronDataset(cfg.dataset_root, phase=phase, start=i,
                          aug_dict=cfg.aug_dict, split_ratio=cfg.split_ratio)

            dataset_dict[phase] = dataset
            dataloader_dict[phase] = DataLoader((dataset), batch_size=cfg.batch_size,
                                                shuffle=cfg.shuffle, num_workers=cfg.num_workers)

        dataset_list.append(dataset_dict)
        dataloader_list.append(dataloader_dict)

    dataset = {}

    dataloader = {}

    dataset['test'] = NeuronDataset(cfg.dataset_root, phase='test',
                             aug_dict=cfg.aug_dict, split_ratio=cfg.split_ratio)

    dataloader['test'] = DataLoader(dataset['test'], batch_size=cfg.batch_size,
                          shuffle=cfg.shuffle, num_workers=cfg.num_workers)





else:

    dataset = {
        phase: NeuronDataset(cfg.dataset_root, phase=phase,
                             aug_dict=cfg.aug_dict, split_ratio=cfg.split_ratio)
        for phase in ['train', 'val', 'test']
    }

    dataloader = {
        phase: DataLoader(dataset[phase], batch_size=cfg.batch_size,
                          shuffle=cfg.shuffle, num_workers=cfg.num_workers)
        for phase in ['train', 'val', 'test']
    }


# 6.初始化criterion.
criterion = get_criterion(cfg)

# 7.初始化optimizer.
optimizer = get_optimizer(cfg)(model.parameters(), **cfg.optim_param[cfg.optim_name])

# 8.初始化scheduler.
def set_scheduler():
    return get_scheduler(cfg)(optimizer, **cfg.shdlr_param[cfg.shdlr_name])

# 9.初始化meter.
loss_meter = AverageValueMeter()
score_meter = ConfusionMeter(cfg.num_class['out'])
# model_stat_meter = AverageValueMeter()


