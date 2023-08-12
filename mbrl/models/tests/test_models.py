import tempfile

import torch

from .models import DNNTestModel, PNNTestModel
from mbrl.models import EnsembleModel
from omegaconf import OmegaConf


def ensemble(input_width, output_width, ensemble_size, deterministic):
    model_cfg = OmegaConf.empty()
    if deterministic:
        model_cfg.clazz = 'mbrl.models.tests.models.DNNTestModel'
    else:
        model_cfg.clazz = 'mbrl.models.tests.models.PNNTestModel'

    return EnsembleModel(input_width, output_width, 'cpu', ensemble_size, model_cfg)


def test_dnn_output():
    batch_size = 16
    input_width = 4
    output_width = 6
    model = DNNTestModel(input_width, output_width, 'cpu')
    output = model.forward(torch.empty((batch_size, input_width)).uniform_())
    assert output.size() == (batch_size, output_width)


def test_pnn_output():
    batch_size = 16
    input_width = 4
    output_width = 6
    model = PNNTestModel(input_width, output_width, 'cpu')
    output = model.forward(torch.empty((batch_size, input_width)).uniform_())
    assert output.size() == (batch_size, output_width, 2)


def test_save_load():
    batch_size = 16
    input_width = 4
    output_width = 6
    ensemble_size = 1
    model = ensemble(input_width, output_width, ensemble_size, False)
    test_input = torch.empty(batch_size, input_width).uniform_()
    output1 = model.forward(test_input)
    with tempfile.TemporaryFile() as f:
        torch.save(model, f)
        f.seek(0)
        loaded_model = torch.load(f)
        output2 = loaded_model.forward(test_input)
        assert ((output1 - output2).abs().sum() < 1e-6)
