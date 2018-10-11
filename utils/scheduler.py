
from torch.optim.lr_scheduler import StepLR

key2scheduler = {
    'StepLR': StepLR
}

def get_scheduler(cfg):

    assert cfg.shdlr_name in key2scheduler, \
        '{} is not a key of key2scheduler.'.format(cfg.shdlr_name)

    assert cfg.shdlr_name != None, \
        'Scheduler unsure.'

    return key2scheduler[cfg.shdlr_name]
