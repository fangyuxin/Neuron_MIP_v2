import time
import copy
from init import *
from tqdm import tqdm



def _train_and_val(dataloader, get_model=False):

    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_score = {metric: 0 for metric in cfg.metrics}
    model.module.reset()
    scheduler = set_scheduler()

    for epoch in range(cfg.num_epochs):

        since = time.time()

        print('Epoch {}/{}'.format(epoch + 1, cfg.num_epochs))
        print('-' * 50)

        if get_model:
            phase_list = ['train']
        else:
            phase_list = ['train', 'val']

        for phase in phase_list:

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

                    vis_pred = pred.float()
                    vis_target = target.float()

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

                        vis.img('Ori', input[0])
                        vis.img('pred', vis_pred[0])
                        vis.img('label', vis_target[0])



            epoch_loss = loss_meter.get_value()[0]
            epoch_IoU = score_meter.get_scores('IoU')
            epoch_Dice = score_meter.get_scores('Dice')
            print('{} Loss: {:.4f}   IoU: {:.4f}   Dice: {:.4f}'.\
                format(phase, epoch_loss, epoch_IoU, epoch_Dice))

            time_elapsed = time.time() - since
            print('Epoch_{} complete in {:.0f}m {:.0f}s'. \
                  format(epoch, time_elapsed // 60, time_elapsed % 60))

            if (phase == 'val' or get_model == True) \
                    and epoch_IoU > best_score['IoU'] \
                    and epoch_Dice > best_score['Dice']:
                best_model_wts = copy.deepcopy(model.module.state_dict())
                best_score['IoU'] = epoch_IoU
                best_score['Dice'] = epoch_Dice

    return model, best_score, best_model_wts



def _train_and_val_CV():


    for k in range(cfg.k_fold_split):

        print('Fold: {}/{}'.format(k + 1, cfg.k_fold_split))

        model, best_score, best_model_wts = _train_and_val(dataloader_list[k])

        for metric in best_score:
            if k == 0:
                best_score_dict = {metric: AverageValueMeter() for metric in best_score}

            best_score_dict[metric].update(best_score[metric])

    for metric in best_score_dict:
        best_score_dict[metric] = best_score_dict[metric].get_value()[0]
        print('{}: {}  '.format(metric, best_score_dict[metric]))

    return model, best_score_dict, best_model_wts


def get_model(best_score=None):

    dataset = {
        phase: NeuronDataset(cfg.dataset_root, phase=phase,
                             aug_dict=cfg.aug_dict, split_ratio=0)
        for phase in ['train']
    }

    dataloader = {
        phase: DataLoader(dataset[phase], batch_size=cfg.batch_size,
                          shuffle=cfg.shuffle, num_workers=cfg.num_workers)
        for phase in ['train']
    }

    model, _, best_model_wts = _train_and_val(dataloader, get_model=True)

    if best_score:
        best_model_info = ''
        for metric in best_score:
            best_model_info += '{}_{:.4f}_'.format(metric, best_score[metric])

    else:
         best_model_info = best_score

    model.module.load_state_dict(best_model_wts)
    model.module.save(best_model_info)



def train_and_val():

    if cfg.CV:
        _, best_score, _ = _train_and_val_CV()

    else:
        _, best_score, _ = _train_and_val(dataloader)

    get_model(best_score)




get_model(None)

# train_and_val()



