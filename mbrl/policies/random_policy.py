import numpy as np
import torch

from mbrl.optimizers import ActionSequence
from . import Policy


class RandomPolicy(Policy):
    def __init__(self, device, action_space, planning_horizon):
        self.device = device
        self.action_space = action_space
        self.planning_horizon = planning_horizon

    def plan_action_sequence(self, state) -> ActionSequence:
        a = ActionSequence()
        a.actions = np.stack(
            [np.random.uniform(self.action_space.low, self.action_space.high) for _ in range(self.planning_horizon)])

        # TODO: generate using torch
        a.actions = torch.from_numpy(a.actions).to(dtype=torch.float32, device=self.device)
        return a

    def reset_trajectory(self):
        pass
