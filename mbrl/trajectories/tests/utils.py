import torch

from mbrl import utils
from mbrl.dynamics_model import *
from mbrl.models import *
from omegaconf import OmegaConf


class TestDModel(TorchModel):
    def __init__(self, input_width, output_width, device):
        super(TestDModel, self).__init__(device)
        self.input_width = input_width
        self.output_width = output_width
        self.value = -1

    def forward(self, x):
        return torch.Tensor(x.size(0), self.output_width).fill_(self.value)

    def is_deterministic(self):
        return True


class TestPModel(TorchModel):
    def __init__(self, input_width, output_width, device):
        super(TestPModel, self).__init__(device)
        self.input_width = input_width
        self.output_width = output_width
        self.value = -1
        self.variance = 0

    def forward(self, x):
        batch_size = x.size(0)
        mean = torch.Tensor(batch_size, self.output_width).fill_(self.value)
        var = torch.Tensor(batch_size, self.output_width).fill_(self.variance)
        return torch.stack((mean, var), dim=2)

    def is_deterministic(self):
        return False


def create_cfg():
    cfg = OmegaConf.empty()
    cfg.device = 'cpu'
    cfg.env = {}
    cfg.env.state_size = 1
    cfg.env.action_size = 1
    cfg.env.state_transformer = 'mbrl.environments.hooks.default_state_transformer'
    cfg.env.target_transformer = 'mbrl.environments.hooks.DefaultTargetTransformer'
    return cfg


def create_nn_dynamics_model(cfg, model):
    return NNBasedDynamicsModel(jit=False,
                                model=model,
                                trainer=None,
                                device=cfg.device,
                                state_size=cfg.env.state_size,
                                action_size=cfg.env.action_size,
                                state_transformer=cfg.env.state_transformer,
                                target_transformer=cfg.env.target_transformer)


def get_deterministic_dynamics_model(cfg):
    model = OmegaConf.empty()
    model.clazz = utils.fullname(TestDModel)
    return create_nn_dynamics_model(cfg, model)


def get_probabilistic_dynamics_model(cfg):
    model = OmegaConf.empty()
    model.clazz = utils.fullname(TestPModel)
    return create_nn_dynamics_model(cfg, model)


def get_deterministic_ensemble_dynamics_model(cfg, ensemble_size):
    model = OmegaConf.empty()
    model.clazz = utils.fullname(EnsembleModel)
    model.params = {}
    model.params.ensemble_size = ensemble_size
    model.params.model = {}
    model.params.model.clazz = utils.fullname(TestDModel)
    return create_nn_dynamics_model(cfg, model)
