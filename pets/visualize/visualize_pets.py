import os
import sys

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())

from mbrl import AbstractMain
from collections import defaultdict
import argparse
from omegaconf import OmegaConf
import torch
import pets.visualize as vis
import glob
import os
from mbrl import utils
from natsort import natsorted
import visdom


class VisualizePets(AbstractMain):

    def start(self):
        args, cfg = self.get_args_and_cfg()
        self.vis_obj = visdom.Visdom(port=args.port)
        if args.command == 'visualize':
            self.visualize(args.file, args.vis_env, args.delta, args.sort)
        elif args.command == 'visualize_group':
            self.visualize_group(args.logs, args.file, args.vis_env, filter_regex=None)
        elif args.command == 'visualize_sweep':
            self.visualize_sweep(args.dir, args.vis_env, args.filter)
        elif args.command == 'visualize_dynamics':
            self.visualize_dynamics(args.file, args.vis_env)
        else:
            raise ValueError(f"Unknown command {args.command}")

    def get_args(self):
        parser = argparse.ArgumentParser(description='Visualize PETS checkpoints')
        parser.add_argument('--port', help='Visdom Port', default=1234)
        parser.add_argument('--vis_env', help='Visdom environment name')
        parser.add_argument('--config', '-c', help='Main config file name', default='pets/conf/config.yaml')
        parser.add_argument('--log_config', '-l', help='Log configuration file', default='pets/conf/logging.yaml')
        parser.add_argument('--verbose', '-v',
                            help='Activate debug logging, if "all" is specified will '
                                 'activate for root logger,  otherwise takes a comma separated list of loggers',
                            nargs='?',
                            default='')

        subparsers = parser.add_subparsers(help='sub-command help', dest='command')

        visualize_parser = subparsers.add_parser('visualize', help='Visualize a log file')
        visualize_parser.add_argument('--file', '-f', help='Checkpoint file', type=utils.readable_file)
        visualize_parser.add_argument('--sort', '-s', help='Sort by ground truth Y values', type=utils.boolean_string,
                                      default=True)
        visualize_parser.add_argument('--delta', '-d', help='Display deltas', type=utils.boolean_string,
                                      default=True)

        visualize_group_parser = subparsers.add_parser('visualize_group',
                                                       help='Visualize a group of jobs in a directory')
        visualize_group_parser.add_argument('--logs',
                                            '-d',
                                            nargs='*',
                                            help='Group dir', type=utils.readable_dir,
                                            required=True)
        visualize_group_parser.add_argument('--file',
                                            '-f',
                                            help='Optional checkpoint file to load from all directories,'
                                                 'otherwise teh latest checkpoint is loaded from each dir',
                                            type=str,
                                            default=None)

        visualize_sweep_parser = subparsers.add_parser('visualize_sweep',
                                                       help='Visualize a jobs under a sweep output directory')
        visualize_sweep_parser.add_argument('--dir', '-d', help='Sweep directory', type=utils.readable_dir)
        visualize_sweep_parser.add_argument('--filter',
                                            '-f',
                                            help='Regex filter for filtering matching groups',
                                            type=str,
                                            default=None)

        visualize_dynamics_parser = subparsers.add_parser('visualize_dynamics',
                                                          help='Visualize a set of dynamics model tests from a given checkpoint')
        visualize_dynamics_parser.add_argument('--file', '-f', help='Checkpoint File', type=utils.readable_file)

        return parser.parse_args()

    def visualize(self, file, vis_env, delta, sort):
        full_conf = OmegaConf.load("{}/config.yaml".format(os.path.dirname(os.path.realpath(file))))
        vis_log = torch.load(file)
        vis.test_and_visualize(self.vis_obj, vis_env, full_conf, vis_log.get('trial_num', 'unknown'), vis_log,
                               delta=delta, sort=sort)
        vis.visualize_reward(self.vis_obj, vis_env, vis_log)
        vis.visualize_loss(self.vis_obj, vis_env, vis_log)

    def visualize_group(self, logs_dirs, file, vis_env, filter_regex):
        self.log.info(f"Visualizing job group in {logs_dirs}")
        logs = defaultdict(list)
        configs = defaultdict(list)

        def load_log(directory, trial_file=None):
            full_conf = OmegaConf.load(f"{directory}/config.yaml")
            trial_files = glob.glob(f"{directory}/trial_*.dat")
            if len(trial_files) > 1:
                if trial_file is not None:
                    last_trial_log = f"{directory}/{trial_file}"
                else:
                    last_trial_log = max(trial_files, key=os.path.getctime)
                self.log.info(f"Loading {last_trial_log}")
                vis_log = torch.load(last_trial_log)
                logs[log_dir].append(vis_log)
                configs[log_dir].append(full_conf)

        for log_dir in logs_dirs:
            if os.path.exists(os.path.join(log_dir, 'config.yaml')):
                self.log.info(f"Loading latest trial from {log_dir}")
                d = os.path.join(log_dir)
                load_log(d)
            else:
                # Assuming directory with multiple identical experiments (dir/0, dir/1 ..)
                latest = defaultdict(list)
                for ld in os.listdir(log_dir):
                    directory = os.path.join(log_dir, ld)
                    if os.path.isdir(directory):
                        trial_files = glob.glob(f"{directory}/trial_*.dat")
                        if len(trial_files) == 0:
                            continue
                        last_trial_log = max(trial_files, key=os.path.getctime)
                        last_trial_log = last_trial_log[len(directory) + 1:]
                        latest[log_dir].append(last_trial_log)

                for ld in os.listdir(log_dir):
                    if ld == '.slurm': continue
                    log_subdir = os.path.join(log_dir, ld)
                    if os.path.isdir(log_subdir):
                        if file is None:
                            # Load data for the smallest trial number from all sub directories
                            if len(latest[log_dir]) == 0:
                                self.log.warn(f"No trial files found under {log_dir}")
                                break
                            trial_file = natsorted(latest[log_dir])[0]
                        else:
                            trial_file = file
                        load_log(log_subdir, trial_file)
        self.log.info(f"Loaded logs from {len(logs)} directories")
        vis.visualize_group(self.vis_obj, vis_env, logs, configs, filter_regex)

    def visualize_sweep(self, sweep_dir, vis_env, filter_regex):
        dirs = []
        for ld in os.listdir(sweep_dir):
            d = os.path.join(sweep_dir, ld)
            if os.path.isdir(d) and ld != '_slurm_logs_':
                dirs.append(d)
        self.visualize_group(dirs, None, vis_env, filter_regex)

    def visualize_dynamics(self, checkpoint_file, vis_env):
        self.log.info(f"Visualizing dynamics model in in {checkpoint_file}")
        logs = defaultdict(list)
        configs = defaultdict(list)

        checkpoint_dir = checkpoint_file[:checkpoint_file.rfind("/") + 1]
        # TODO generate the needed configs as done in visualize_group
        if os.path.exists(os.path.join(checkpoint_dir, 'config.yaml')):
            full_conf = OmegaConf.load(f"{checkpoint_dir}/config.yaml")
            self.log.info(f"Loading {checkpoint_file}")
            vis_log = torch.load(checkpoint_file)
            logs[checkpoint_file].append(vis_log)
            configs[checkpoint_file].append(full_conf)
        else:
            raise ValueError("Checkpoint directory missing expected config")

        vis.visualize_dynamics(self.vis_obj, vis_env, logs, configs)


if __name__ == '__main__':
    VisualizePets()
