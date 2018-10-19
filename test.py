
from init import *
from config import *
from utils import *
from tqdm import tqdm
import imageio




def _test(dataloader):

    model.module.load('./checkpoints/Unet_Dice_86_ds=8192')

    phase = 'test'

    model.eval()
    loss_meter.reset()
    score_meter.reset()

    list2csv(['result_name', 'IoU', 'Dice', 'Acc', 'Recall'], './result_stat.csv', mode='w+')

    for i, (input, target) in tqdm(enumerate(dataloader[phase], start=0), total=len(dataloader[phase])):

        input = input.to(device)
        target = target.to(device)

        with torch.no_grad():

            output = model(input)

            pred = cfg.softmax(output)
            pred = torch.max(pred, dim=1)[1]

            score_meter.update(output, target)

            vis_pred = pred.float()
            vis_target = target.float()

            for j in range(cfg.test_batch_size):
                result_stat = [score for score in score_meter.get_scores(is_single=True).values()]

                result = torch.cat(((input[j][0] * 255).byte(), (vis_target[j] * 255).byte(), (vis_pred[j] * 255).byte()), dim=1)
                result_name = ('result_{}.png').format(i * cfg.test_batch_size + j)
                result_stat.insert(0, result_name)

                list2csv(result_stat, cfg.csv_path)

                imageio.imwrite(cfg.result_path + result_name, result.cpu().numpy())


            if i % cfg.print_freq == 0:

                vis.plot('IoU', score_meter.get_scores('IoU'))
                vis.plot('Dice', score_meter.get_scores('Dice'))

                vis.img('Result', result)


    epoch_IoU = score_meter.get_scores('IoU')
    epoch_Dice = score_meter.get_scores('Dice')

    print('{}  IoU: {:.4f}   Dice: {:.4f}'.\
        format(phase, epoch_IoU, epoch_Dice))

    return model


_test(dataloader)







