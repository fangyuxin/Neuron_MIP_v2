
from config import *
import models
from data import *
from utils import *


def init(**kwargs):
    cfg.parse(kwargs)

# 1.初始化config, 更新相关参数.
init()

# 2.初始化visdom环境.
vis = Visualizer(cfg.env)

# 3.初始化device.
device = cfg.device

# 4.初始化model.
model = getattr(models, cfg.model) \
    (cfg.num_class['in'], cfg.num_class['out']).to(device)
model = nn.DataParallel(model)

# 5.初始化dataset, dataloader, augmentation.

dataset = {
    phase: NeuronDataset(cfg.dataset_root, phase=phase,
                         aug_dict=cfg.aug_dict)
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
scheduler = get_scheduler(cfg)(optimizer, **cfg.shdlr_param[cfg.shdlr_name])

# 9.初始化meter.
loss_meter = AverageValueMeter()
score_meter = ConfusionMeter(cfg.num_class['out'])




