import logging
import torch
import torch.cuda
import torch.nn as nn

from mbrl import utils
from mbrl.models.torchmodel import TorchModel

log = logging.getLogger(__name__)


class EnsembleModel(TorchModel):

    def __init__(self, input_size, output_size, device, ensemble_size, model):
        super(EnsembleModel, self).__init__(device)

        def model_factory():
            return utils.instantiate(model, input_size, output_size, device)

        self.ensemble_size = ensemble_size
        self.models = nn.ModuleList([model_factory().to(device=device) for _ in range(ensemble_size)])

    def get_ensemble_size(self):
        return self.ensemble_size

    def get_models(self):
        return self.models

    def forward(self, x):
        """
        DE(x[batch, input_dim]) -> y[batch, output_dim, ensemble_size]
        PE(x[batch, input_dim]) -> y[batch, output_dim, ensemble_size, 2], where y[:, :. :, 0] is mean and y[:, :. :, 1] is variance
        :param x: input
        """
        assert torch.is_tensor(x)
        assert x.dim() == 2
        outputs = [self.models[eid](x) for eid in range(self.ensemble_size)]
        return torch.stack(outputs, dim=2)

    def get_input_size(self):
        return self.models[0].get_input_size()

    def get_output_size(self):
        return self.models[0].get_output_size()

    def is_ensemble(self):
        return True

    def is_deterministic(self):
        return self.models[0].is_deterministic()

    def is_probabilistic(self):
        return self.models[0].is_probabilistic()
