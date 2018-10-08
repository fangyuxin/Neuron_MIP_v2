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


    def reset(self):
        self.cf.fill(0)


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

            self.cf += np.bincount(position, minlength=self.num_class ** 2). \
                reshape(self.num_class, self.num_class)


    def get_scores(self, metrics=None):

        cf = self.cf

        IoU = np.diag(cf) / (np.sum(cf, axis=0) + np.sum(cf, axis=1) - np.diag(cf))
        mean_IoU = np.nanmean(IoU)

        acc = np.diag(cf).sum() / cf.sum()

        scores = {
            'IoU': mean_IoU,
            'Acc': acc

        #     ...

        }

        if metrics == None:
            for m in scores:
                print(m + ': {}   '.format(scores[m]))

        else:
            print(str(metrics) + ': {}   '.format(scores[str(metrics)]))

        return scores















'''
以下代码作测试用，经初步检验，我写的ConfusionMatrix应该没有错。

'''





# class runningScore(object):
#     def __init__(self, n_classes):
#         self.n_classes = n_classes
#         self.confusion_matrix = np.zeros((n_classes, n_classes))
#
#     def _fast_hist(self, label_true, label_pred, n_class):
#         mask = (label_true >= 0) & (label_true < n_class)
#         hist = np.bincount(
#             n_class * label_true[mask].astype(int) + label_pred[mask],
#             minlength=n_class ** 2,
#         ).reshape(n_class, n_class)
#         return hist
#
#     def update(self, label_trues, label_preds):
#         for lt, lp in zip(label_trues, label_preds):
#             self.confusion_matrix += self._fast_hist(
#                 lt.flatten(), lp.flatten(), self.n_classes
#             )
#
#     def get_scores(self):
#         """Returns accuracy score evaluation result.
#             - overall accuracy
#             - mean accuracy
#             - mean IU
#             - fwavacc
#         """
#         hist = self.confusion_matrix
#         acc = np.diag(hist).sum() / hist.sum()
#         acc_cls = np.diag(hist) / hist.sum(axis=1)
#         acc_cls = np.nanmean(acc_cls)
#         iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
#         mean_iu = np.nanmean(iu)
#         freq = hist.sum(axis=1) / hist.sum()
#         fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#         cls_iu = dict(zip(range(self.n_classes), iu))
#
#         return acc
#
#     def reset(self):
#         self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
#

# N, C, d1, d2 = 5, 4, 16, 16
#
# p1 = torch.empty(N).random_(0, C)
# p2 = torch.rand(N, C)
# p3 = torch.empty(N, d1, d2).random_(0, C)
# p4 = torch.empty(N, d1, d2).random_(0, C).long()
#
# t1 = torch.empty(N).random_(0, C)
# t2 = torch.rand(N, C)
# t3 = torch.empty(N, d1, d2).random_(0, C).long()
# t4 = torch.empty(N, C, d1, d2).random_(0, 2)
#
# P = [p4]
# T = [t3]
#
# m1 = ConfusionMeter(C)
# m2 = runningScore(C)
#
# for p, t in zip(P, T):
#     m1.update(p, t)
#     m1.get_scores('Acc')
#     m1.reset()
#
#     # m2.add(p, t)
#     # v = m2.value()
#     # print('acc: {}'.format(np.diag(v).sum() / v.sum()))
#     # m2.reset()
#
#     m2.update(p.cpu().numpy(), t.cpu().numpy())
#     print('acc: {}'.format(m2.get_scores()))










    # '''Test'''
    #
    # running_score = runningScore(2)
    #
    # label_true = np.array([[0,0,1],
    #               [1,1,0],
    #               [0,1,0]])
    #
    # label_pred = np.array([[0,0,1],
    #               [1,1,0],
    #               [0,1,0]])
    #
    # running_score.update(label_true, label_pred)
    #
    # print(running_score.get_scores())
    #



# class CM():
#     """Maintains a confusion matrix for a given calssification problem.
#
#     The ConfusionMeter constructs a confusion matrix for a multi-class
#     classification problems. It does not support multi-label, multi-class problems:
#     for such problems, please use MultiLabelConfusionMeter.
#
#     Args:
#         k (int): number of classes in the classification problem
#         normalized (boolean): Determines whether or not the confusion matrix
#             is normalized or not
#
#     """
#
#     def __init__(self, k, normalized=False):
#
#         self.conf = np.ndarray((k, k), dtype=np.int32)
#         self.normalized = normalized
#         self.k = k
#         self.reset()
#
#     def reset(self):
#         self.conf.fill(0)
#
#     def add(self, predicted, target):
#         """Computes the confusion matrix of K x K size where K is no of classes
#
#         Args:
#             predicted (tensor): Can be an N x K tensor of predicted scores obtained from
#                 the model for N examples and K classes or an N-tensor of
#                 integer values between 0 and K-1.
#             target (tensor): Can be a N-tensor of integer values assumed to be integer
#                 values between 0 and K-1 or N x K tensor, where targets are
#                 assumed to be provided as one-hot vectors
#
#         """
#         predicted = predicted.cpu().numpy()
#         target = target.cpu().numpy()
#
#         assert predicted.shape[0] == target.shape[0], \
#             'number of targets and predicted outputs do not match'
#
#         if np.ndim(predicted) != 1:
#             assert predicted.shape[1] == self.k, \
#                 'number of predictions does not match size of confusion matrix'
#             predicted = np.argmax(predicted, 1)
#         else:
#             assert (predicted.max() < self.k) and (predicted.min() >= 0), \
#                 'predicted values are not between 1 and k'
#
#         onehot_target = np.ndim(target) != 1
#         if onehot_target:
#             assert target.shape[1] == self.k, \
#                 'Onehot target does not match size of confusion matrix'
#             assert (target >= 0).all() and (target <= 1).all(), \
#                 'in one-hot encoding, target values should be 0 or 1'
#             # assert (target.sum(1) == 1).all(), \
#             #     'multi-label setting is not supported'
#             target = np.argmax(target, 1)
#         else:
#             assert (predicted.max() < self.k) and (predicted.min() >= 0), \
#                 'predicted values are not between 0 and k-1'
#
#         # hack for bincounting 2 arrays together
#         x = predicted + self.k * target
#         bincount_2d = np.bincount(x.astype(np.int32),
#                                   minlength=self.k ** 2)
#         assert bincount_2d.size == self.k ** 2
#         conf = bincount_2d.reshape((self.k, self.k))
#
#         self.conf += conf
#
#     def value(self):
#         """
#         Returns:
#             Confustion matrix of K rows and K columns, where rows corresponds
#             to ground-truth targets and columns corresponds to predicted
#             targets.
#         """
#         if self.normalized:
#             conf = self.conf.astype(np.float32)
#             return conf / conf.sum(1).clip(min=1e-12)[:, None]
#         else:
#             return self.conf
#
















