import gym
import numpy
import torch

# noinspection PyUnresolvedReferences
from mbrl import environments
from mbrl.dynamics_model import EnvBasedDynamicsModel

def predict_batch_impl_test(batch_size, num_workers):
    """
    Testing prediction of a batch of actions and states
    """
    # make sure that predict_batch outputs the expected results.
    # This is useful to test optimizations in the batched implementation
    env_name = 'MBRLCartpole-v0'
    envs = [gym.make(env_name) for _ in range(batch_size)]
    s0 = [envs[i].reset() for i in range(batch_size)]
    s0 = numpy.vstack(s0)
    actions = [envs[i].action_space.sample() for i in range(batch_size)]
    actions = numpy.vstack(actions)
    s1 = [envs[i].step(actions[i])[0] for i in range(batch_size)]

    dm = EnvBasedDynamicsModel(env_name, num_workers)
    next_predictions = dm.predict(torch.from_numpy(s0), torch.from_numpy(actions))
    assert torch.is_tensor(next_predictions)
    next_predictions = next_predictions.numpy()
    [numpy.testing.assert_array_equal(next_predictions[i], s1[i]) for i in range(batch_size)]
    assert (next_predictions.shape[0] == batch_size)


def test_predict_batch():
    predict_batch_impl_test(1, 10)
    predict_batch_impl_test(20, 2)
    predict_batch_impl_test(40, 10)
    predict_batch_impl_test(400, 10)
