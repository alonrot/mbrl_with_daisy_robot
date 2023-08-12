import torch.nn as nn

from mbrl.models import AbstractDNN
from mbrl.models import AbstractPNN


class DNNTestModel(AbstractDNN):
    def __init__(self, input_width, output_width, device):
        self.input_width = input_width
        w = 500
        act = nn.Tanh
        layers = nn.Sequential(
            nn.Linear(input_width, w),
            act(),
            nn.Linear(w, w),
            act(),
            nn.Linear(w, w),
            act(),
            nn.Linear(w, output_width),
        )
        super(DNNTestModel, self).__init__(input_width, output_width, layers, device)


class PNNTestModel(AbstractPNN):
    def __init__(self, input_width, output_width, device):
        self.input_width = input_width
        w = 500
        act = nn.Tanh
        layers = nn.Sequential(
            nn.Linear(input_width, w),
            act(),
            nn.Linear(w, w),
            act(),
            nn.Linear(w, w),
            act(),
        )
        super(type(self), self).__init__(input_width, output_width, layers, w, device)
