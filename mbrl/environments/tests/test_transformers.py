import pytest
import torch

from mbrl import utils

transformations = [
    # # no not transform target
    # ('environments.hooks.NoOpTargetTransformer.forward', 'environments.hooks.NoOpTargetTransformer.reverse'),
    # # # generic transformations, used by most environments
    # ('environments.hooks.DefaultTargetTransformer.forward', 'environments.hooks.DefaultTargetTransformer.reverse'),
    # half cheetah
    ('mbrl.environments.half_cheetah.TargetTransformer.forward',
     'mbrl.environments.half_cheetah.TargetTransformer.reverse'),
]

inputs = [
    # 1d input, a single state pair
    (torch.Tensor([10, 20, 30, 40]), torch.Tensor([1, 2, 3, 4])),
    # 2d input, a vector of state pairs
    (torch.Tensor([[10, 20, 30, 40]]), torch.Tensor([[1, 2, 3, 4]])),
]


def test_target_transformation():
    """
    Tests that the target transformer pairs are correctly reversing
    given s0, s1:

    T = transformer(s0, s1)
    reverse(s0, T) == s1
    """

    for str_transformer, str_reverse in transformations:
        transformer = utils.get_static_method(str_transformer)
        reverse = utils.get_static_method(str_reverse)
        for s0, s1 in inputs:
            new_s1 = transformer(s0, s1)
            postproc_s1 = reverse(s0, new_s1)
            assert s1.numpy() == pytest.approx(postproc_s1.numpy())
