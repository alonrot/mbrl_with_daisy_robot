import torch
import torch.nn as nn

from mbrl.models.torchmodel import TorchModel


class AbstractPNN(TorchModel):
    """
    Probabilistic network neural network
    """

    def __init__(self, input_size, output_size, layers, w, device):
        super(AbstractPNN, self).__init__(device)
        self.layers = layers
        self.input_size = input_size
        self.output_size = output_size
        self.mean = nn.Linear(w, output_size)
        self.var = nn.Sequential(
            nn.Linear(w, output_size),
            nn.Softplus()
        )
        self.to(self.device)

    def forward(self, x):
        """
        Predicts mean and variance
        :param x: input [batch X input_size]
        :return: output [batch X output_size X 2]
                 where output[:,:,0] is the mean and output[:,:,1] is the variance
        """
        assert x.dtype == torch.float32
        y = self.layers(x)
        return torch.stack((self.mean(y), self.var(y)), dim=2)

    def get_input_size(self):
        return self.input_size

    def get_output_size(self):
        return self.output_size

    def is_deterministic(self):
        return False
