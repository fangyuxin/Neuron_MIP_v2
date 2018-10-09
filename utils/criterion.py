import torch.nn as nn


key2loss = {
    'BCELoss': nn.BCELoss,
    'CrossEntropyLoss': nn.CrossEntropyLoss
}


def get_criterion(cfg):

    assert cfg.loss_name not in key2loss, \
        '{} is not a key of key2loss.'.format(cfg.loss_name)

    if cfg.loss_name == None:
        return nn.CrossEntropyLoss()

    elif cfg.loss_param == None:
        return key2loss[cfg.loss_name]()

    else:
        return key2loss[cfg.loss_name](**cfg.loss_param)








