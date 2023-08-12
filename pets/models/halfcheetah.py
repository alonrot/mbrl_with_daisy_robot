import torch.nn as nn

from mbrl.models import AbstractDNN
from mbrl.models import AbstractPNN


class HalfCheetahDNN(AbstractDNN):
    def __init__(self, input_width, output_width, device):
        self.input_width = input_width
        w = 200
        act = nn.Tanh
        layers = nn.Sequential(
            nn.Linear(input_width, w),
            act(),
            nn.Linear(w, w),
            act(),
            nn.Linear(w, w),
            act(),
            nn.Linear(w, w),
            act(),
            nn.Linear(w, output_width)
        )
        super(HalfCheetahDNN, self).__init__(input_width, output_width, layers, device)



class HalfCheetahPNN(AbstractPNN):
    def __init__(self, input_width, output_width, device):
        self.input_width = input_width
        w = 200
        act = nn.Tanh
        layers = nn.Sequential(
            nn.Linear(input_width, w),
            act(),
            nn.Linear(w, w),
            act(),
            nn.Linear(w, w),
            act(),
            nn.Linear(w, w),
            act(),
        )
        super(type(self), self).__init__(input_width, output_width, layers, w, device)
