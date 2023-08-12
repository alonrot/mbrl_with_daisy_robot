from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
from gym import utils
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/half_cheetah.xml' % dir_path, 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        reward = HalfCheetahEnv.get_reward(ob, action)

        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            (self.sim.data.qpos[:1] - self.prev_qpos[:1]) / self.dt,
            self.sim.data.qpos[1:],
            self.sim.data.qvel,
        ])

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(loc=0, scale=0.001, size=self.model.nq)
        qvel = self.init_qvel + np.random.normal(loc=0, scale=0.001, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.sim.data.qpos)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55

    @staticmethod
    def preprocess_state_np(state):
        assert isinstance(state, np.ndarray)
        assert state.ndim in (1, 2)
        d1 = state.ndim == 1
        if d1:
            # if input is 1d, expand it to 2d
            state = np.expand_dims(state, 0)
        # [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.] ->
        # [1., sin(2), cos(2)., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.]
        ret = np.concatenate([state[:, 1:2], np.sin(state[:, 2:3]), np.cos(state[:, 2:3]), state[:, 3:]], axis=1)
        if d1:
            # and squeeze it back afterwards
            ret = ret.squeeze()
        return ret

    @staticmethod
    def preprocess_state(state):
        assert torch.is_tensor(state)
        assert state.dim() in (1, 2)
        d1 = state.dim() == 1
        if d1:
            # if input is 1d, expand it to 2d
            state = state.unsqueeze(0)
        ret = torch.cat([state[:, 1:2], torch.sin(state[:, 2:3]), torch.cos(state[:, 2:3]), state[:, 3:]], dim=1)
        if d1:
            # and squeeze it back afterwards
            ret = ret.squeeze()
        return ret

    @staticmethod
    def get_reward(next_ob, action):
        """
        :param next_ob: the new state we got to
        :param action:  the action that led to this state
        :return: the reward for the transition
        """
        assert isinstance(next_ob, np.ndarray)
        assert isinstance(action, np.ndarray)
        assert next_ob.ndim in (1, 2)

        was1d = next_ob.ndim == 1
        if was1d:
            next_ob = np.expand_dims(next_ob, 0)
            action = np.expand_dims(action, 0)

        reward_ctrl = -0.1 * np.square(action).sum(axis=1)
        reward_run = next_ob[:, 0] - 0.0 * np.square(next_ob[:, 2])
        reward = reward_run + reward_ctrl

        if was1d:
            reward = reward.squeeze()
        return reward

    @staticmethod
    def get_reward_torch(next_ob, action):
        assert torch.is_tensor(next_ob)
        assert torch.is_tensor(action)
        assert next_ob.dim() in (1, 2)
        next_ob = next_ob.float()
        action = action.float()

        was1d = next_ob.dim() == 1
        if was1d:
            next_ob = next_ob.unsqueeze(0)
            action = action.unsqueeze(0)

        reward_ctrl = -0.1 * (action * action).sum(dim=1)
        reward_run = next_ob[:, 0] - 0.0 * (next_ob[:, 2] * next_ob[:, 2])
        reward = reward_run + reward_ctrl

        if was1d:
            reward = reward.squeeze()

        return reward

    def compute_next_state(self, state, action):
        """
        Computes the the state the environment will get to if it starts at state and takes the given action.
        Note: This changes the current state.
        """
        # This is tricky because the state includes the velocity at state[0], and that is a function of
        # the previous state and the current state.
        # I am not sure we can do it within the current framework
        raise NotImplemented("This function is not implemented yet")

    def compute_next_states(self, states, actions):
        """
        For each pair of state and action in the given input arrays:
        Computes the the state the environment will get to if starts at state takes the given action.
        Note: This changes the current state.
        returns a corresponding array for the new states
        """
        # This is tricky because the state includes the velocity at state[0], and that is a function of
        # the previous state and the current state.
        # I am not sure we can do it within the current framework
        raise NotImplemented("This function is not implemented yet")


class TargetTransformer:
    @staticmethod
    def forward(s0, s1):
        assert torch.is_tensor(s0) and torch.is_tensor(s1)
        if s0.dim() == 1:
            return TargetTransformer.forward(s0.unsqueeze(0), s1.unsqueeze(0)).squeeze()
        return torch.cat([s1[:, :1], s1[:, 1:] - s0[:, 1:]], dim=1)

    @staticmethod
    def reverse(s0, predicted_s1):
        assert torch.is_tensor(s0) and torch.is_tensor(predicted_s1)
        if s0.dim() == 1:
            return TargetTransformer.reverse(s0.unsqueeze(0), predicted_s1.unsqueeze(0)).squeeze()
        return torch.cat([predicted_s1[:, :1], s0[:, 1:] + predicted_s1[:, 1:]], dim=1)
