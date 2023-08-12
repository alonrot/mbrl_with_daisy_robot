import gym.envs.registration as registration
import gym.error

import mbrl.environments.hooks

# Making special environment DaisyRealRobotEnv visible to pets/main.py:
# NOTE alonrot: DaisyRealRobotEnv is not a gym environment, and thus, shouldn't be registered below.
from mbrl.environments.daisy_real import DaisyRealRobotEnv

try:
    registration.register(
        id='MBRLCartpole-v0',
        entry_point='mbrl.environments.cartpole:CartpoleEnv'
    )

    registration.register(
        id='MBRLReacher3D-v0',
        entry_point='mbrl.environments.reacher:Reacher3DEnv'
    )

    registration.register(
        id='MBRLPusher-v0',
        entry_point='mbrl.environments.pusher:PusherEnv'
    )

    registration.register(
        id='MBRLHalfCheetah-v0',
        entry_point='mbrl.environments.half_cheetah:HalfCheetahEnv'
    )

    registration.register(
        id='PyBulletReacher-v0',
        entry_point='mbrl.environments.pybullet_envs:PyBulletReacher'
    )

    registration.register(
        id='PyBulletSawyer-v0',
        entry_point='mbrl.environments.pybullet_envs:PyBulletSawyer'
    )

    registration.register(
        id='PyBulletKuka-v0',
        entry_point='mbrl.environments.pybullet_envs:PyBulletKuka'
    )
    registration.register(
        id='Daisy-v0',
        entry_point='mbrl.environments.daisy:DaisyEnv'
    )
except gym.error.Error as e:
    # this module may be initialized multiple times. ignore gym registration errors.
    pass
