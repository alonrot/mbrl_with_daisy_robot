import torch.nn as nn

from .abstractmodel import AbstractModel


class TorchModel(AbstractModel, nn.Module):

    def __init__(self, device):
        super(TorchModel, self).__init__()
        self.device = device

    def get_device(self):
        return self.device
