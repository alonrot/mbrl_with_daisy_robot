from time import time

import torch
from mbrl import utils
from mbrl.dynamics_model import DynamicsModel


def compute_trajectories_and_returns_batched(model,
                                             state0,
                                             actions,
                                             reward_func,
                                             batch_size,
                                             compute_trajectories_func,
                                             return_traj=False): # added alonrot
    """
    :param model: dynamics model to use
    :param state0: initial state
    :param actions: actions matrix [num_trajectories X planning_horizon X action_size]
    :param reward_func:
    :param batch_size:
    :param compute_trajectories_func:
    :param return_traj:
    :return: a tupple (returs, trajectories (None unless return_traj==True) , timing1, timing2)
    """
    assert callable(compute_trajectories_func)
    assert callable(reward_func)
    assert actions.size(0) % batch_size == 0
    assert actions.dim() == 3
    num_batches = actions.size(0) // batch_size

    # inherit device from input data, otherwise it fails
    assert state0.device == actions.device
    device = state0.device

    returns = torch.zeros(actions.size(0), dtype=torch.float32, device=device)
    total_compute_traj_time = 0
    total_compute_return_time = 0

    num_trajectories = actions.shape[0]
    planning_horizon = actions.shape[1]
    state_size = state0.size(0)
    all_trajectories = None
    if return_traj:
        # all trajectories may not fit on GPU, copy one batch at a time to CPU.
        all_trajectories = torch.zeros((num_trajectories, planning_horizon + 1, state_size),
                                       dtype=state0.dtype,
                                       device=device) # alonrot changed: before it was device="cpu"

    for batch_num in range(num_batches):
        t = time()
        action_batch = actions.narrow(0, batch_num * batch_size, batch_size)
        batch_trajectories = compute_trajectories_func(model, state0, action_batch)

        if return_traj:
            all_trajectories.narrow(0, batch_num * batch_size, batch_size).copy_(batch_trajectories)

        total_compute_traj_time += (time() - t)
        t = time()
        batch_returns = compute_returns(batch_trajectories, action_batch, reward_func) # original
        
        # NOTE: narrow() https://kite.com/python/docs/torch.FloatTensor.narrow
        returns.narrow(0, batch_num * batch_size, batch_size).copy_(batch_returns)
        total_compute_return_time += (time() - t)

    return returns, all_trajectories, total_compute_traj_time, total_compute_return_time


def compute_returns(trajectories, actions, reward_function):
    assert actions.dim() == 3
    num_trajectories = actions.shape[0]
    planning_horizon = actions.shape[1]
    returns = torch.zeros(num_trajectories, dtype=trajectories.dtype, device=trajectories.device)
    for timestep in range(planning_horizon):
        actions_batch = actions[:, timestep, None]
        # remove singleton dimension (the column width is always 1)
        actions_batch = actions_batch.reshape((actions_batch.shape[0], actions_batch.shape[2]))
        reward = reward_function(trajectories[:, timestep + 1], actions_batch)  # trajectories [num_trajectories x (planning_horizon+1) x state_size]
                                                                                # trajectories[:, timestep + 1].shape = [num_trajectories x state_size]
        assert reward is not None
        assert reward.size() == returns.size()
        returns += reward
    assert not torch.isnan(returns).any(), "NaNs detected in returns, aborting"
    return returns

def compute_trajectories(model: DynamicsModel, state0: torch.TensorType, actions: torch.TensorType):
    """
    Computes state trajectory for state0 and a matrix of actions
    :param model: dynamics model used to compute s1 from s0, a0
    :param state0: state 0 for all trajectories
    :param actions: 3 dimensional matrix of actions (num_trajectories, planning_horizon, action_size)
    :return: trajectories tensor on the same device a the input states, with size [actions.size(0), actions.size(1) + 1]
             representing [trajectories X horizon + 1], the first column trajectories[:, 0] is all state0
    """
    assert torch.is_tensor(state0)
    assert torch.is_tensor(actions)
    assert actions.dim() == 3
    num_trajectories = actions.size(0)
    planning_horizon = actions.size(1)
    state_size = state0.size(0)
    trajectories = torch.zeros((num_trajectories,
                                planning_horizon + 1,
                                state_size),
                               dtype=state0.dtype,
                               device=state0.device)
    deterministic = model.is_deterministic()

    trajectories[:, 0] = state0.repeat(num_trajectories).view(num_trajectories, state_size)
    for trajectory_idx in range(planning_horizon):
        state_batch = trajectories[:, trajectory_idx]
        actions_batch = actions[:, trajectory_idx, None]
        # remove singleton dimension (the column width is always 1)
        actions_batch = actions_batch.reshape((actions_batch.shape[0], actions_batch.shape[2]))
        output = model.predict(state_batch, actions_batch)

        if model.is_ensemble():
            # simple propagation cannot handle ensembles directly.
            # the ensemble prediction is transformed to that of a single model (D or P):
            # DE -> D (Mean of predictions)
            # PE -> P (Moment matching)
            if model.is_deterministic():
                # DE combine predictions with mean
                output = torch.mean(output, dim=2)
            elif model.is_probabilistic():
                # PE combine predictions with MM
                mean, var = utils.moment_matching_torch(output[:, :, :, 0], output[:, :, :, 1])
                output = torch.stack([mean, var], dim=2)

        if deterministic:
            new_states = output
            trajectories[:, trajectory_idx + 1] = new_states
        else:
            mean = output[:, :, 0]
            variance = output[:, :, 1]
            stddev = torch.sqrt(variance)
            new_states = torch.normal(mean, stddev)

            trajectories[:, trajectory_idx + 1] = new_states
    return trajectories

def gather_actions(optimizer, N, s0, da):
    actions = torch.empty([optimizer.planning_horizon, N, da])
    # generate N episodes
    for n in range(N):
        optimizer.reset_trajectory()
        action_sequence = optimizer.plan_action_sequence(s0)
        actions[:, n, :] = action_sequence.actions
    return actions
