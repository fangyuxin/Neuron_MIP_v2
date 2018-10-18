import time
from init import *
from config import *
from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.misc as misc
import imageio

# def find_best_model(model_root='./checkpoints'):
#
#     model_list = [os.path.join(model_root, model_name) \
#                   for model_name in os.listdir(model_root)]
#
#     best_model_name = sorted(model_list, key=lambda x: \
#                         float(x.split(':')[-1]) + \
#                         float(x.split(':')[-2].split('b')[0]))[-1]
#
#     return best_model_name




def _test(dataloader):

    model.module.load('./checkpoints/Unet_IoU_0.7687_Dice_0.8533_')

    phase = 'test'

    model.eval()
    loss_meter.reset()
    score_meter.reset()

    list2csv(['result_name', 'IoU', 'Dice', 'Acc', 'Recall'], './result_stat.csv', mode='w+')

    for i, (input, target) in tqdm(enumerate(dataloader[phase], start=0), total=len(dataloader[phase])):
    # for i, (input, target) in enumerate(dataloader[phase], start=1):

        input = input.to(device)
        target = target.to(device)

        with torch.no_grad():

            output = model(input)

            pred = cfg.softmax(output)
            pred = torch.max(pred, dim=1)[1]

            # for j in range(cfg.batch_size):
            #     misc.imsave('./all_dataset/ori_{}.png'.format(i + j), pred[i].numpy())
            #     misc.imsave('./all_dataset/pred_{}.png'.format(i + j), pred[i].numpy())
            #     misc.imsave('./all_dataset/label_{}.png'.format(i + j), pred[i].numpy())

            score_meter.update(output, target)

            vis_pred = pred.float()
            vis_target = target.float()

            for j in range(cfg.batch_size):
                result_stat = [score for score in score_meter.get_scores(is_single=True).values()]

                result = torch.cat(((input[j][0] * 255).byte(), (vis_target[j] * 255).byte(), (vis_pred[j] * 255).byte()), dim=1)
                result_name = ('result_{}.png').format(i * cfg.batch_size + j)
                result_stat.insert(0, result_name)

                list2csv(result_stat, './result_stat.csv')

                imageio.imwrite(cfg.dataset_root + '/result/' + result_name, result.cpu().numpy())


                # imageio.imwrite('./all_dataset/pred/ori_{}.png'.format(i + j), input[i][0].cpu().numpy())
                # imageio.imwrite('./all_dataset/pred/pred_{}.png'.format(i + j), vis_pred[i].cpu().numpy())
                # imageio.imwrite('./all_dataset/pred/label_{}.png'.format(i + j), vis_target[i].cpu().numpy())

            if i % cfg.print_freq == 0:

                vis.plot('IoU', score_meter.get_scores('IoU'))
                vis.plot('Dice', score_meter.get_scores('Dice'))

                vis.img('Result', result)

                # vis.img('Ori', input[0])
                # vis.img('pred', vis_pred[0])
                # vis.img('label', vis_target[0])


    epoch_IoU = score_meter.get_scores('IoU')
    epoch_Dice = score_meter.get_scores('Dice')

    print('{}  IoU: {:.4f}   Dice: {:.4f}'.\
        format(phase, epoch_IoU, epoch_Dice))

    return model


_test(dataloader)







