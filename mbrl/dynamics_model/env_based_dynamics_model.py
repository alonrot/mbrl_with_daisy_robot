import logging
import math
from scipy.optimize import approx_fprime
from itertools import islice

import gym
import numpy as np
import torch

from mbrl.dynamics_model import DynamicsModel
from mbrl.dynamics_model.parallel import SubprocVecEnv

log = logging.getLogger(__name__)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


# A cheating model, what if we had the best possible model of the environment?
class EnvBasedDynamicsModel(DynamicsModel):

    def __init__(self, env_name, num_workers=40):
        supported = {
            'MBRLCartpole-v0',
            'PyBulletReacher-v0',
            'PyBulletSawyer-v0',
            'PyBulletKuka-v0'
        }
        assert env_name in supported, \
            "EnvBasedDynamicsModel supported environments: %s : provided : %s" % (supported, env_name)

        self.env_name = env_name
        self.num_envs = num_workers
        self.env = None
        self.subprocEnv = None
        self.initialize_env()

    def initialize_env(self):
        def make_env(seed):
            def _():
                env = gym.make(self.env_name)
                env.seed(seed)
                return env

            return _

        start_seed = 0
        inits = [make_env(s) for s in range(start_seed, start_seed + self.num_envs)]
        self.subprocEnv = SubprocVecEnv(inits)
        self.env = gym.make(self.env_name)

    def __getstate__(self):
        d = self.__dict__.copy()
        del d['subprocEnv']
        del d['env']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        self.initialize_env()

    def predict(self, states: torch.TensorType, actions: torch.TensorType) -> torch.TensorType:
        """Predicts a batch of states/action transitions"""
        if isinstance(states, np.ndarray):
            log.warning('Numpy input in EnvBasedDynamicsModel.predict')
            states = torch.from_numpy(states)
            actions = torch.from_numpy(actions)

        assert torch.is_tensor(states)
        assert torch.is_tensor(actions)
        assert states.dim() == 2
        assert actions.dim() == 2
        assert states.size(0) == actions.size(0)

        # underlying environment works with numpy arrays, convert from Torch to numpy
        states = states.cpu().numpy()
        actions = actions.cpu().numpy()

        num_states = states.shape[0]
        num_envs = self.subprocEnv.num_envs
        assert states.ndim == 2
        assert actions.ndim == 2
        assert num_states == actions.shape[0]

        batch_size = math.ceil(num_states / num_envs)
        padding = num_envs * batch_size - num_states

        states = np.pad(states, ((0, padding), (0, 0)), 'constant')
        actions = np.pad(actions, ((0, padding), (0, 0)), 'constant')
        state_chunks = list(chunk(states, batch_size))
        actions_chunks = list(chunk(actions, batch_size))
        obs_chunks = self.subprocEnv.compute_next_states_batch(state_chunks, actions_chunks)
        obs = np.concatenate(obs_chunks)
        obs = obs[0: len(obs) - padding]

        return torch.from_numpy(obs)

    def predict_with_uncertainty(self, state, action):
        """
        Return the same as self.predict, but with an uncertainty variance estimate. 0 if deterministic model (default).
        :return: (prediction, variance estimate)
        """
        prediction = self.predict(state, action)
        return prediction, np.zeros_like(prediction)

    def dx_fd(self, state, action, eps=1e-8):
        """
        Default finite difference d predict / d state
        """
        return np.vstack([
            approx_fprime(state, lambda state: self.predict(state, action).numpy()[i], eps)
            for i in range(len(state))
        ])

    def du_fd(self, state, action, eps=1e-8):
        """
        Default finite difference d predict / d action
        """
        return np.vstack([
            approx_fprime(action, lambda action: self.predict(state, action).numpy()[i], eps)
            for i in range(len(state))
        ])

    def is_ensemble(self):
        return False

    def is_deterministic(self):
        return True

