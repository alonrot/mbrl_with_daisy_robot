from mbrl.trajectories import *
from mbrl.trajectories.tests import utils
import torch


def _test_ts_prop_deterministic_ensemble(particles, trajectories, horizon, ensemble_size, permute_assignment):
    cfg = utils.create_cfg()
    dm = utils.get_deterministic_ensemble_dynamics_model(cfg, ensemble_size)
    dm.nn.value = 1

    state0 = torch.Tensor(cfg.env.state_size).fill_(0.5)
    actions = torch.Tensor(trajectories, horizon, cfg.env.action_size)
    for t in range(trajectories):
        actions[t, :, :].fill_(t)

    def reward(_next_ob, action):
        return action.sum(dim=1)

    prop = TSProp(particles=particles, return_trajectories=True, permute_assignment=permute_assignment)
    traj = prop.compute_trajectories_and_returns(dm, state0, actions, reward)

    for trajectory in range(trajectories):
        assert traj.all_returns[trajectory] == actions[trajectory].sum()


def test_ts_prop_deterministic():
    _test_ts_prop_deterministic_ensemble(particles=2,
                                         trajectories=2,
                                         horizon=3,
                                         ensemble_size=1,
                                         permute_assignment=True)

    _test_ts_prop_deterministic_ensemble(particles=2,
                                         trajectories=2,
                                         horizon=3,
                                         ensemble_size=2,
                                         permute_assignment=True)

    _test_ts_prop_deterministic_ensemble(particles=3,
                                         trajectories=5,
                                         horizon=3,
                                         ensemble_size=5,
                                         permute_assignment=True)

    _test_ts_prop_deterministic_ensemble(particles=2,
                                         trajectories=4,
                                         horizon=3,
                                         ensemble_size=2,
                                         permute_assignment=False)

    _test_ts_prop_deterministic_ensemble(particles=20,
                                         trajectories=2500,
                                         horizon=30,
                                         ensemble_size=5,
                                         permute_assignment=False)
