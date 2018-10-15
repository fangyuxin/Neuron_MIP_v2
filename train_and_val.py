
import copy
from init import *
from tqdm import tqdm



def train_and_val():

    # 训练.

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_score = {'best_IoU': 0, 'best_Dice': 0}

    for epoch in range(cfg.num_epochs):

        print('Epoch {}/{}'.format(epoch + 1, cfg.num_epochs))
        print('-' * 50)

        # 每一个epoch有2个phase: train, val.
        for phase in ['train', 'val']:

            if phase == 'train':
                scheduler.step()
                model.train()

            if phase == 'val':
                model.eval()

            loss_meter.reset()
            score_meter.reset()

            for i, (input, target) in tqdm(enumerate(dataloader[phase], start=1),
                                           total=len(dataloader[phase])):

                input = input.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):

                    output = model(input)
                    loss = criterion(output, target)

                    pred = cfg.softmax(output)
                    pred = torch.max(pred, dim=1)[1]

                    vis_pred = pred[0].float()
                    vis_target = target[0].float()

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

            if phase == 'val' \
                    and epoch_IoU > best_score['best_IoU'] \
                    and epoch_Dice > best_score['best_Dice']:
                best_model_wts = copy.deepcopy(model.module.state_dict())
                best_score['best_IoU'] = epoch_IoU
                best_score['best_Dice'] = epoch_Dice


    best_model_info = ''
    for metric in best_score:
        best_model_info += '{}: {:.4f}  '.format(metric, best_score[metric])

    model.module.load_state_dict(best_model_wts)
    model.module.save(best_model_info)
    return model


train_and_val()








