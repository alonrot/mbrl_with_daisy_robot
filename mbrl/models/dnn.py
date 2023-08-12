# Deterministic NN

import torch

from mbrl.models.torchmodel import TorchModel


class AbstractDNN(TorchModel):

    def __init__(self, input_size, output_size, layers, device):
        super(AbstractDNN, self).__init__(device)
        self.layers = layers
        self.input_size = input_size
        self.output_size = output_size
        self.to(self.device)

    def forward(self, x):
        x = x.to(device=self.device, dtype=torch.float32)
        return self.layers(x)

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size

    def is_deterministic(self):
        return True
