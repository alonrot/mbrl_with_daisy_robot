import logging

import torch

from mbrl import utils
from mbrl.policies import Policy

log = logging.getLogger(__name__)


class PETSPolicy(Policy):

    def __init__(self, cfg, planning_horizon, traj, optimizer):
        self.cfg = cfg
        self.device = cfg.device
        self.planning_horizon = planning_horizon
        self.trajectory_propagator_cfg = traj
        self.trajectory_propagator = None
        self.optimizer_cfg = optimizer
        self.optimizer = None
        self.model = None
        self.reward_func = None

    def setup(self, model, action_space, reward_func):
        self.model = model
        self.reward_func = reward_func
        self.optimizer = utils.instantiate(self.optimizer_cfg,
                                           self.device,
                                           action_space,
                                           self.planning_horizon)
        self.optimizer.setup(self.compute_return)
        self.trajectory_propagator = utils.instantiate(self.trajectory_propagator_cfg)
        log.info(f"MPC optimizer: {self.optimizer}")
        log.info(f"MPC trajectory propagator: {self.trajectory_propagator}")

    def plan_action_sequence(self, state0):
        return self.optimizer.plan_action_sequence(state0)

    def reset_trajectory(self):
        self.optimizer.reset_trajectory()

    def compute_return(self, state0, actions):
        assert torch.is_tensor(state0)
        assert torch.is_tensor(actions)
        state0 = state0.to(device=self.device)

        # alonrot: mbrl.trajectories.tsprop.TSProp.compute_trajectories_and_returns()
        return self.trajectory_propagator.compute_trajectories_and_returns(self.model, # alonrot: top-level dynamics model
                                                                           state0,
                                                                           actions,
                                                                           self.reward_func)

    def __repr__(self):
        return "PETSPolicy"

class PETSPolicyParametrized(PETSPolicy):
    """
    Whe using a parametrized policy, the action space becomes a 'parameter space'.
    The main difference is parameter_space in .setup()

    author: alonrot
    """

    def __init__(self, cfg, planning_horizon, traj, optimizer):
        super().__init__(cfg, planning_horizon, traj, optimizer)

    def setup(self, model, parameter_space, reward_func):
        self.model = model
        self.reward_func = reward_func

        self.optimizer = utils.instantiate(self.optimizer_cfg,
                                           self.device,
                                           parameter_space,
                                           self.planning_horizon)
        self.optimizer.setup(self.compute_return)
        self.trajectory_propagator = utils.instantiate(self.trajectory_propagator_cfg)
        log.info(f"MPC optimizer: {self.optimizer}")
        log.info(f"MPC trajectory propagator: {self.trajectory_propagator}")

    def plan_action_sequence(self, state0, t_curr):
        return self.optimizer.plan_action_sequence(state0, t_curr)

    def __repr__(self):
        return "PETSPolicyParametrized"


