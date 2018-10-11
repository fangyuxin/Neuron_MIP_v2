
import copy
import torch
from utils import *
from config import *
import models
import torch.nn as nn
from data import *
from utils import *
from tqdm import *
from torchnet import meter as M


def train_and_val(**kwarg):


    # 1.初始化config, 更新相关参数.
    cfg.parse(kwarg)

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
        phase: NeuronDataset(cfg.dataset_root, phase=phase, \
                               aug_dict=cfg.aug_dict)
        for phase in ['train', 'val']
    }

    dataloader = {
        phase: DataLoader(dataset[phase], batch_size=cfg.batch_size, \
                          num_workers=cfg.num_workers)
        for phase in ['train', 'val']
    }


    # 6.初始化criterion.
    criterion = get_criterion(cfg)

    # 7.初始化optimizer.
    optimizer = get_optimizer(cfg)(model.parameters(), \
                                   **cfg.optim_param[cfg.optim_name])

    # 8.初始化scheduler.
    scheduler = get_scheduler(cfg)(optimizer, \
                                   **cfg.shdlr_param[cfg.shdlr_name])

    # 9.初始化meter.
    loss_meter = AverageValueMeter()
    score_meter = ConfusionMeter(cfg.num_class['out'])



    # 训练.

    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = 0.0

    for epoch in range(cfg.num_epochs):

        print('Epoch {}/{}'.format(epoch + 1, cfg.num_epochs))
        print('-' * 10)

        # 每一个epoch有2个phase: train, val.
        for phase in ['train', 'val']:

            if phase == 'train':
                scheduler.step()
                model.train()

            if phase == 'val':
                model.eval()

            loss_meter.reset()
            score_meter.reset()

            for i, (input, target) in enumerate(dataloader[phase], start=1):

                input = input.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    output = model(input)
                    loss = criterion(output, target)
                    # print(output.size())

                    pred = cfg.softmax(output)
                    pred = torch.max(pred, dim=1)[1]

                    vis_pred = pred[0].float()
                    vis_target = target[0].float()

                    # pred_count1 = (pred == 1).long()
                    # print(pred.sum())


                    # print(pred.size())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                with torch.no_grad():

                    loss_meter.update(loss.item())
                    score_meter.update(output, target)

                    if i % cfg.print_freq == 0:

                        vis.plot('loss', loss_meter.get_value()[0])
                        vis.plot('IoU', score_meter.get_scores('IoU'))
                        vis.plot('Dice', score_meter.get_scores('Dice'))

                        vis.img('pred', vis_pred)
                        vis.img('label', vis_target)


            epoch_loss = loss_meter.get_value()[0]
            epoch_IoU = score_meter.get_scores('IoU')
            epoch_Dice = score_meter.get_scores('Dice')
            print('{} Loss: {:.4f}   IoU: {:.4f}   Dice: {:.4f}'.\
                format(phase, epoch_loss, epoch_IoU, epoch_Dice))

            if phase == 'val' and epoch_IoU > best_score:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_score = epoch_IoU


    model.load_state_dict(best_model_wts)
    model.save()
    return model



train_and_val()








