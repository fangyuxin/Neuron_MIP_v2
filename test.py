import os
import copy
from init import *
from config import *
from utils import *
from tqdm import tqdm


def find_best_model(model_root='./checkpoints'):

    model_list = [os.path.join(model_root, model_name) \
                  for model_name in os.listdir(model_root)]

    best_model_name = sorted(model_list, key=lambda x: \
                        float(x.split(':')[-1]) + \
                        float(x.split(':')[-2].split('b')[0]))[-1]

    return best_model_name




def _test():

    # 训练.
    model.module.load(find_best_model())

    phase = 'test'

    model.eval()
    loss_meter.reset()
    score_meter.reset()

    for i, (input, target) in tqdm(enumerate(dataloader[phase], start=1), total=len(dataloader[phase])):

        input = input.to(device)
        target = target.to(device)

        with torch.no_grad():

            output = model(input)

            pred = cfg.softmax(output)
            pred = torch.max(pred, dim=1)[1]

            score_meter.update(output, target)

            vis_pred = pred[0].float()
            vis_target = target[0].float()

            if i % cfg.print_freq == 0:

                vis.plot('IoU', score_meter.get_scores('IoU'))
                vis.plot('Dice', score_meter.get_scores('Dice'))

                vis.img('pred', vis_pred)
                vis.img('label', vis_target)


    epoch_loss = loss_meter.get_value()[0]
    epoch_IoU = score_meter.get_scores('IoU')
    epoch_Dice = score_meter.get_scores('Dice')

    print('{} Loss: {:.4f}   IoU: {:.4f}   Dice: {:.4f}'.\
        format(phase, epoch_loss, epoch_IoU, epoch_Dice))

    return model


_test()







