import logging
from time import time

import torch

from mbrl.optimizers import ActionSequence
from mbrl.trajectories import utils
from .trajectory_propagator import TrajectoryPropagator

log = logging.getLogger(__name__)


class TSProp(TrajectoryPropagator):
    def __init__(self, particles, return_trajectories, permute_assignment):
        """
        :param particles:
        :param return_trajectories:
        :param permute_assignment: if true, we will be permuting the particle->model assignment
                                   at every time step (TS1), otherwise not (TSInf)
        """
        self.particles = particles
        self.return_trajectories = return_trajectories

        # TSInf: each particle stays with a single model throughout the trajectory.
        # TS1 : each particle samples a model from the ensemble at each time step
        self.permute_assignment = permute_assignment

    def __str__(self):
        return f"TSProp ({'TS1' if self.permute_assignment else 'TSInf'})"

    def compute_trajectories_and_returns(self, model, state0, actions, reward_func) -> ActionSequence:
        assert torch.is_tensor(state0)
        assert torch.is_tensor(actions)
        assert actions.dim() == 3
        assert model.is_ensemble()
        assert self.particles >= 1

        total_time = time()

        num_action_sequences = actions.size(0) # Ntrajectories
        planning_horizon = actions.size(1)
        action_size = actions.size(2)
        state_size = state0.size(0)
        ensemble_size = model.get_ensemble_size()
        num_trajectories = num_action_sequences * self.particles
        assert num_trajectories % ensemble_size == 0
        num_trajectories_per_model = num_trajectories // ensemble_size

        repl_action = actions.repeat(1, self.particles, 1)
        repl_action = repl_action.view(num_trajectories, -1, action_size)

        # assignment of propagation to model
        assignment = torch.arange(0,
                                  ensemble_size,
                                  dtype=torch.float32,
                                  device=actions.device).repeat(num_trajectories_per_model)

        trajectories = torch.zeros((num_trajectories,
                                    planning_horizon + 1,
                                    state_size),
                                   dtype=state0.dtype,
                                   device=state0.device)

        deterministic = model.is_deterministic()

        # first column is state0
        trajectories[:, 0] = state0.repeat(num_trajectories).view(num_trajectories, state_size)

        traj_time = 0
        ret_time = 0
        ret = ActionSequence()
        t = time()        
        for timestep in range(planning_horizon):
            for eid in range(ensemble_size):
                particles_for_ensemble_index = assignment == eid
                assert particles_for_ensemble_index.sum() == num_trajectories_per_model

                # Get batches:
                state_batch = trajectories[:, timestep][particles_for_ensemble_index]
                action_batch = repl_action[:, timestep][particles_for_ensemble_index]

                # Predict:
                output = model.predict_one_model(state_batch, action_batch, eid)

                # Update:
                model_idx = particles_for_ensemble_index.nonzero().view(-1)
                if deterministic:
                    new_states = output
                    trajectories[:, timestep + 1][model_idx] = new_states
                else:
                    mean = output[:, :, 0]
                    variance = output[:, :, 1]
                    stddev = torch.sqrt(variance)
                    new_states = torch.normal(mean, stddev)

                    trajectories[:, timestep + 1][model_idx] = new_states

            if self.permute_assignment:
                assignment = assignment[torch.randperm(assignment.nelement())]
        traj_time += (time() - t)

        t = time()

        # NOTE: compute_returns() just adds up all the reward signals, which are computed using
        # whatever reward_func is specified
        # returns is a vector, with the accumulated reward per trajectory (i.e., particle of the ensemble)
        returns = utils.compute_returns(trajectories, repl_action, reward_func)
        ret_time += (time() - t)

        total_time = time() - total_time
        log.debug("compute_trajectories_and_returns timing "
                  f"(total={total_time:.3f}s traj_time=({traj_time:.3f}, ret_time={ret_time:.3f}))")

        # group results by particle and average across each particle
        ret.all_returns = returns.view(num_action_sequences, self.particles).mean(dim=1)
        ret.all_actions = actions
        ret.all_returns_with_particles = returns.view(num_action_sequences, self.particles) # alonrot added: For curiosity exploration

        # alonrot added: returning the entire trajectories as well
        if self.return_trajectories:
            if trajectories is not None:
                trajectories = trajectories.view(num_action_sequences,
                                                 self.particles,
                                                 planning_horizon + 1,
                                                 state0.size(0))
            ret.all_trajectories = trajectories
        return ret
