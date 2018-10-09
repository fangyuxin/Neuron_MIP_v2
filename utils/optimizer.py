
from torch.optim import SGD
from torch.optim import Adam
from torch.optim import RMSprop


key2optim = {
    'SGD': SGD,
    'Adam': Adam,
    'RMSprop': RMSprop
}

def get_optimizer(cfg):

    assert cfg.optim_name not in key2optim, \
        '{} is not available.'.format(cfg.optim_name)

    if cfg.optim_name == None:
        return SGD

    elif:
        return key2optim[cfg.optim_name]


