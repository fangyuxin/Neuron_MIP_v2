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
        if name == None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

        torch.save(self.state_dict(), name)
        return name
