import torch
import numpy as np


'''
    Confusion Matrix（混淆矩阵） 可用于快速生成各个评价指标。
    这里的混淆矩阵可以接受的参数类型比较多。
    predicted和target的形式可为：
        1.(N)
        2.(N, C)
        3.(N, d1, d2, ..., dk)
        4.(N, C, d1, d2, ..., dk)

'''

class ConfusionMeter():

    def __init__(self, num_class):

        self.num_class = num_class

        self.cf = np.ndarray((num_class, num_class), dtype=np.int32).reshape(num_class, num_class)
        self.reset()

        self.one_cf = np.ndarray((num_class, num_class), dtype=np.int32).reshape(num_class, num_class)
        self.one_reset()


    def reset(self):
        self.cf.fill(0)

    def one_reset(self):
        self.one_cf.fill(0)


    def process(self, predicted, target):

        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()
        batch_size = predicted.shape[0]

        if np.ndim(predicted) != 1 and np.ndim(predicted) != 3:
            predicted = np.argmax(predicted, axis=1).reshape(batch_size, -1)
        else:
            predicted = predicted.reshape(batch_size, -1)

        if np.ndim(target) != 1 and np.ndim(target) != 3:
            target = np.argmax(target, axis=1).reshape(batch_size, -1)
        else:
            target = target.reshape(batch_size, -1)

        return predicted.astype(np.int32), target.astype(np.int32)


    def update(self, predicted, target):

        predicted, target = self.process(predicted, target)

        for p, t in zip(predicted, target):
            position = t * self.num_class + p

            self.one_cf = np.bincount(position, minlength=self.num_class ** 2). \
                reshape(self.num_class, self.num_class)

            self.cf += self.one_cf


    def get_scores(self, metrics=None, is_single=False):

        if is_single:
            cf = self.one_cf
        else:
            cf = self.cf

        IoU = np.diag(cf) / (np.sum(cf, axis=0) + np.sum(cf, axis=1) - np.diag(cf))
        mean_IoU = np.nanmean(IoU)

        dice = 2 * np.diag(cf) / (np.sum(cf, axis=0) + np.sum(cf, axis=1))
        mean_dice = np.nanmean(dice)

        acc = np.diag(cf).sum() / cf.sum()

        recall = np.diag(cf) / np.sum(cf, axis=1)
        mean_recall = np.nanmean(recall)

        scores = {
            'IoU': mean_IoU,
            'Dice': mean_dice,
            'Acc': acc,
            'Recall': mean_recall

            # ...

        }

        if metrics == None:
            return scores

        else:
            return scores[metrics]





class AverageValueMeter():
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def update(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def get_value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan
