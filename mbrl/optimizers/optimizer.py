import logging
import numpy as np
import torch

from mbrl.dataset import SASDataset

log = logging.getLogger(__name__)


class ActionSequence:
    def __init__(self):
        # matrix of evaluated action sequences
        self.all_actions = None
        # state trajectories generated using the dynamics model with s0 and above actions
        self.all_trajectories = None
        # a list of returns, one for each trajectory and actions sequence
        self.all_returns = None
        # selected action sequence
        self.actions = None
        # return of selected action sequence
        self.seq_return = None

        # TODO alonrot added:
        self.actions_plan_mean = None
        self.actions_plan_var = None
        self.states_plan_mean = None
        self.states_plan_var = None
        self.all_optimized_rewards = None
        self.best_optimized_rewards = None
        self.saturations_count = None
        self.all_returns_with_particles = None # For curiosity exploration

    def get_action(self, x, t):
        # It is critical to copy the action because otherwise it retains a copy to the source torch tensor
        # this will cause a severe leak if that action is stored in a dataset.
        return self.actions[0].clone()

    def get_num_actions(self):
        return len(self.actions)


class Optimizer:
    def plan_action_sequence(self, state0) -> ActionSequence:
        """
        :param state0: state we start planning from
        :return: 
        """
        raise NotImplementedError("Subclass must implement this function")

    def reset_trajectory(self):
        """
        Reset any state associated with the currently computed trajectory
        """
        pass

    def setup(self, *args):
        """
        Configure optimizer
        :param args: variable arguments list, specific subclasses may have different arguments here
        and caller is expected to pass the correct ones.
        """
        pass

    def __str__(self):
        raise NotImplementedError("Subclass must implement this function")
