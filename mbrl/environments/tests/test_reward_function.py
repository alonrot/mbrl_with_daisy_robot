import gym
import numpy as np
import pytest
import torch

# noinspection PyUnresolvedReferences
from mbrl import utils

np_reward_functions = [
    ('MBRLCartpole-v0', 'mbrl.environments.cartpole.CartpoleEnv.get_reward'),
    ('MBRLHalfCheetah-v0', 'mbrl.environments.half_cheetah.HalfCheetahEnv.get_reward'),
]


def test_reward_matching_env():
    """
    Tests that the reward from the given env matches the reward from the static function
    """
    for name, func in np_reward_functions:
        reward_func = utils.get_static_method(func)
        env = gym.make(name)
        for i in range(1000):
            env.reset()
            a = env.action_space.sample()
            s1, r, d, i = env.step(a)
            assert reward_func(s1, a) == pytest.approx(r)


reward_functions_np_vs_torch = [
    ('MBRLCartpole-v0',
     'mbrl.environments.cartpole.CartpoleEnv.get_reward',
     'mbrl.environments.cartpole.CartpoleEnv.get_reward_torch'),
    ('MBRLHalfCheetah-v0',
     'mbrl.environments.half_cheetah.HalfCheetahEnv.get_reward',
     'mbrl.environments.half_cheetah.HalfCheetahEnv.get_reward_torch'),
]


def test_reward_np_vs_torch():
    """
    Tests that the reward from the given env matches the reward from the static function
    """

    for name, np_reward, torch_reward in reward_functions_np_vs_torch:
        np_reward_func = utils.get_static_method(np_reward)
        torch_reward_func = utils.get_static_method(torch_reward)
        env = gym.make(name)
        for i in range(1000):
            s1 = env.reset()
            a = env.action_space.sample()
            reward1 = np_reward_func(s1, a)
            reward2 = torch_reward_func(torch.from_numpy(s1), torch.from_numpy(a))
            assert reward1 == pytest.approx(reward2.numpy())


def test_reward_batched():
    for name, func in np_reward_functions:
        env = gym.make(name)
        reward_func = utils.get_static_method(func)

        n = 1000
        states1 = []
        actions = []
        expected_rewards = []
        for i in range(n):
            env.reset()
            a = env.action_space.sample()
            s1, r, d, i = env.step(a)
            states1.append(s1)
            actions.append(a)
            expected_rewards.append(np.float32(r))
            assert reward_func(s1, a) == pytest.approx(r)

        states1 = np.asarray(states1)
        actions = np.asarray(actions)
        expected_rewards = np.asarray(expected_rewards)
        rewards = np.float32(reward_func(states1, actions))

        for i in range(n):
            assert rewards[i] == pytest.approx(np.asarray(expected_rewards[i]), 0.0001)

        assert isinstance(rewards, np.ndarray)


get_ee_functions = [
    ('MBRLCartpole-v0', 'mbrl.environments.cartpole.CartpoleEnv._get_ee_pos'),
]


def test_get_ee_batched():
    n = 1000
    for name, func in get_ee_functions:
        env = gym.make(name)

        get_ee = utils.get_static_method(func)
        states = np.asarray([env.reset() for _ in range(n)])
        expected_ees = np.asarray([get_ee(states[i]) for i in range(n)])
        batched_ees = get_ee(states)
        for i in range(n):
            assert expected_ees[i] == pytest.approx(batched_ees[i])


envs_test_compute_next_state_matching_env = [
    'MBRLCartpole-v0',
]


def test_compute_next_state_matching_env():
    """
    Tests that the next state computed by env.compute_next_state() matches the one returned by env.step()
    """
    n = 1000
    for name in envs_test_compute_next_state_matching_env:
        env = gym.make(name)
        for i in range(n):
            initial_state = env.reset()
            action = env.action_space.sample()
            expected_state, _, _, _ = env.step(action)
            computed_state = env.compute_next_state(initial_state, action)

            assert expected_state == pytest.approx(computed_state)


preproc_functions = [
    ('MBRLCartpole-v0',
     'mbrl.environments.cartpole.CartpoleEnv.preprocess_state_np',
     'mbrl.environments.cartpole.CartpoleEnv.preprocess_state'),
    ('MBRLHalfCheetah-v0',
     'mbrl.environments.half_cheetah.HalfCheetahEnv.preprocess_state_np',
     'mbrl.environments.half_cheetah.HalfCheetahEnv.preprocess_state'),
]


def test_preprocess_numpy_vs_torch_1d():
    for env_name, np_func, torch_func in preproc_functions:
        env = gym.make(env_name)
        np_state1d = np.arange(0, env.observation_space.shape[0]).astype(np.float32)

        torch_state1d = torch.from_numpy(np_state1d)
        np_func = utils.get_static_method(np_func)
        torch_func = utils.get_static_method(torch_func)
        np_out = np_func(np_state1d)
        torch_out = torch_func(torch_state1d)
        assert np_out == pytest.approx(torch_out.numpy())


def test_preprocess_numpy_vs_torch_2d():
    for env_name, np_func, torch_func in preproc_functions:
        env = gym.make(env_name)
        np_state2d = np.array([
            np.arange(0, env.observation_space.shape[0]).astype(np.float32),
            np.arange(0, env.observation_space.shape[0]).astype(np.float32) + 1
        ])
        torch_state2d = torch.from_numpy(np_state2d)
        np_func = utils.get_static_method(np_func)
        torch_func = utils.get_static_method(torch_func)
        np_out = np_func(np_state2d)
        torch_out = torch_func(torch_state2d)
        assert np_out == pytest.approx(torch_out.numpy())
