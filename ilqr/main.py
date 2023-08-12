import os, sys

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())

import argparse
import gym
from omegaconf import OmegaConf
from mbrl.policies import RandomPolicy
from mbrl.dataset import SASDataset
from mbrl.dynamics_model import EnvBasedDynamicsModel
# noinspection PyUnresolvedReferences
from mbrl import utils, environments
from ilqr.abstract_main import AbstractMain
from ilqr.trajectory_optimization import iLQRPolicy


class iLQRMain(AbstractMain):
    def get_args_and_cfg(self, overrides):
        args, base_cfg = super(iLQRMain, self).get_args_and_cfg()
        cli = OmegaConf.from_cli(args.overrides)
        # merge base config with environment specific config
        env_config = OmegaConf.load(f'ilqr/conf/envs/{args.env}.yaml')
        cfg = OmegaConf.merge(base_cfg, env_config, cli)
        return args, cfg

    def get_args(self):
        parser = argparse.ArgumentParser(description='Model based RL through iLQR trajectory optimization.')
        parser.add_argument('--config', '-c', help='Main config file name', default='ilqr/conf/config.yaml')
        parser.add_argument('--log_config', '-l', help='Log configuration file', default='ilqr/conf/logging.yaml')
        parser.add_argument('--verbose', '-v',
                            help='Activate debug logging, if "all" is specified will '
                                 'activate for root logger,  otherwise takes a comma separated list of loggers',
                            nargs='?',
                            default='')

        parser.add_argument('--env', '-e', required=True, help='Environment name ' '(one of [reacher | sawyer | kuka])')
        parser.add_argument('overrides', nargs='*', help="Any key=value arguments to override config values "
                                                         "(use dots for.nested=overrides)")

        return parser.parse_args()

    def save_config(self, log_dir, cfg):
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as file:
            file.write(cfg.pretty())

    def start(self):
        cfg = self.cfg
        self.log.info("============= Configuration =============")
        self.log.info(f"Config:\n{cfg.pretty()}")
        self.log.info("=========================================")

        self.save_config(cfg.full_log_dir, cfg)

        # Create environment
        env = gym.make(cfg.env.name)
        dX = env.observation_space.shape[0]
        dU = env.action_space.shape[0]

        # Create cost function
        cost = utils.instantiate(cfg.env.cost)

        dynamics_model = EnvBasedDynamicsModel(cfg, num_workers=1)
        policy = iLQRPolicy(dX, dU, dynamics_model, cost, cfg.time_horizon)

        # Motor babbling
        random_policy = RandomPolicy(cfg.device, env.action_space, cfg.time_horizon)
        dataset = SASDataset()
        for trial in range(cfg.motor_babbling.num_trials):
            episode = utils.sample_episode(env, random_policy, cfg.time_horizon).episode
            dataset.add_episode(episode, cfg.device)
        dynamics_model.train(training_dataset=dataset, testing_dataset=None, training_params=None)

        for trial_num in range(cfg.num_trials):
            last_trajectory = dataset[-cfg.time_horizon:]
            policy.optimize(last_trajectory, max_iter=cfg.ilqr.max_iter)

            # Roll out resulting policy
            for rollout in range(cfg.num_rollouts_per_trial):
                episode = utils.sample_episode(env, policy, cfg.time_horizon, cfg.env.render).episode
                dataset.add_episode(episode, cfg.device)

            # Train dynamics data on rollouts
            dynamics_model.train(training_dataset=dataset, testing_dataset=None, training_params=None)


if __name__ == '__main__':
    iLQRMain()
