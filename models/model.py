import time
import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.model_name = str(type(self))

    def load(self, PATH):
        self.load_state_dict(torch.load(PATH))

    def save(self, name=None):

        prefix = 'checkpoints/' + self.model_name.split('.')[-1].split('\'')[0] + '_'

        if name == None:
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

        else:
            name = prefix + name

        torch.save(self.state_dict(), name)
        return name
