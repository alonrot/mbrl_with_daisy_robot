import torch
from torch.distributions import uniform

from . import Optimizer, ActionSequence


class RandomShootingOptimizer(Optimizer):
    def __init__(self,
                 device,
                 action_space,
                 planning_horizon,
                 num_trajectories):
        # compute_return_function is assigned during setup
        self.compute_return_function = None
        self.device = device
        self.planning_horizon = planning_horizon
        self.num_trajectories = num_trajectories
        low = torch.from_numpy(action_space.low).to(device=device)
        high = torch.from_numpy(action_space.high).to(device=device)
        self.uniform = uniform.Uniform(low, high)

    def reset_trajectory(self):
        pass

    def plan_action_sequence(self, state0) -> ActionSequence:
        # Generate num_trajectories by planning_horizon random actions
        actions = self.uniform.sample([self.num_trajectories, self.planning_horizon])

        # evaluate returns for trajectories
        ret = self.compute_return_function(state0, actions)

        # find trajectory with best return
        idx = torch.argmax(ret.all_returns)

        ret.actions = actions[idx]
        ret.seq_return = ret.all_returns[idx]

        return ret

    def setup(self, compute_return_function):
        self.compute_return_function = compute_return_function

    def __str__(self):
        return F"RandomShootingOptimizer (horizon={self.planning_horizon}, num_trajectories={self.num_trajectories})"
