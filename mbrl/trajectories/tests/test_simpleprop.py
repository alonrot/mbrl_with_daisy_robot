from mbrl.trajectories import *
from mbrl.trajectories.tests import utils
import torch


def _test_simple_prop_deterministic(particles, trajectories, horizon, batch_multiplier):
    cfg = utils.create_cfg()
    dm = utils.get_deterministic_dynamics_model(cfg)
    dm.nn.value = 1

    state0 = torch.Tensor(cfg.env.state_size).fill_(0)
    actions = torch.Tensor(trajectories, horizon, cfg.env.action_size)
    for t in range(trajectories):
        actions[t, :, :].fill_(t)

    def reward(_next_ob, action):
        return action.sum(dim=1)

    prop = SimpleProp(batch_multiplier=batch_multiplier, particles=particles, return_trajectories=True)
    traj = prop.compute_trajectories_and_returns(dm, state0, actions, reward)

    for trajectory in range(trajectories):
        assert traj.all_returns[trajectory] == actions[trajectory].sum()


def _test_simple_prop_probablistic(particles, trajectories, horizon, batch_multiplier):
    cfg = utils.create_cfg()
    dm = utils.get_probabilistic_dynamics_model(cfg)
    dm.nn.value = 1
    dm.nn.variance = 0.1

    state0 = torch.Tensor(cfg.env.state_size).fill_(0)
    actions = torch.Tensor(trajectories, horizon, cfg.env.action_size)
    for t in range(trajectories):
        actions[t, :, :].fill_(t)

    def reward(_next_ob, action):
        return action.sum(dim=1)

    prop = SimpleProp(batch_multiplier=batch_multiplier, particles=particles, return_trajectories=True)
    traj = prop.compute_trajectories_and_returns(dm, state0, actions, reward)

    # rewards ignores observations so the test at this point is identical to the deterministic test
    for trajectory in range(trajectories):
        assert traj.all_returns[trajectory] == actions[trajectory].sum()


def test_simple_prop_deterministic():
    _test_simple_prop_deterministic(particles=1, trajectories=30, horizon=3, batch_multiplier=1)
    _test_simple_prop_deterministic(particles=4, trajectories=30, horizon=3, batch_multiplier=2)


def test_simple_prop_probablistic():
    _test_simple_prop_probablistic(particles=1, trajectories=30, horizon=3, batch_multiplier=1)
    _test_simple_prop_probablistic(particles=4, trajectories=30, horizon=3, batch_multiplier=2)
