import logging
from time import time

import torch

import mbrl.trajectories.utils
from mbrl.optimizers import ActionSequence
from .trajectory_propagator import TrajectoryPropagator

log = logging.getLogger(__name__)


class SimpleProp(TrajectoryPropagator):
    def __init__(self, batch_multiplier, particles, return_trajectories):
        self.particles = particles
        self.batch_multiplier = batch_multiplier
        self.return_trajectories = return_trajectories

    def __str__(self):
        return "SimpleProp"

    def compute_trajectories_and_returns(self, model, state0, actions, reward_func) -> ActionSequence:
        assert torch.is_tensor(state0)
        assert torch.is_tensor(actions)
        assert state0.dim() == 1
        assert actions.dim() == 3
        assert state0.device == actions.device

        num_trajectories = actions.size(0)
        planning_horizon = actions.size(1)
        action_size = actions.size(2)

        batch_size = min(num_trajectories * self.batch_multiplier, num_trajectories * self.particles)

        ret = ActionSequence()
        repl_action = actions.repeat(1, self.particles, 1)
        repl_action = repl_action.view(num_trajectories * self.particles, -1, action_size)

        assert repl_action.size(0) % batch_size == 0, f"action={repl_action.size(0)} % batch_size={batch_size} != 0"
        total_time = time()
        returns, trajectories, traj_time, ret_time = \
            mbrl.trajectories.utils.compute_trajectories_and_returns_batched(model,
                                                                             state0,
                                                                             repl_action,
                                                                             reward_func,
                                                                             batch_size,
                                                                             mbrl.trajectories.utils.compute_trajectories,
                                                                             self.return_trajectories)

        if trajectories is not None:
            assert trajectories.device == state0.device
            trajectories = trajectories.view(num_trajectories,
                                             self.particles,
                                             planning_horizon + 1,
                                             state0.size(0))
        total_time = time() - total_time
        log.debug("compute_trajectories_and_returns timing "
                  f"(traj={total_time:.3f}s traj_time=({traj_time:.3f}, ret_time={ret_time:.3f}))")
        # group results by particle and average across each particle
        ret.all_returns = returns.view(-1, self.particles).mean(dim=1)
        ret.all_actions = actions
        if self.return_trajectories:
            ret.all_trajectories = trajectories
        return ret
