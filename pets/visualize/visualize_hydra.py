import os
import sys

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())

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

import hydra
import logging


def visualize(vis_obj, file, vis_env, delta, sort):
    full_conf = OmegaConf.load("{}/config.yaml".format(os.path.dirname(os.path.realpath(file))))
    vis_log = torch.load(file)
    vis.test_and_visualize(vis_obj, vis_env, full_conf, vis_log.get('trial_num', 'unknown'), vis_log, delta=delta,
                           sort=sort)
    vis.visualize_reward(vis_obj, vis_env, vis_log)
    vis.visualize_loss(vis_obj, vis_env, vis_log)


def visualize_group(vis_obj, logs_dirs, file, vis_env, filter_regex):
    log.info(f"Visualizing job group in {logs_dirs}")
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
            log.info(f"Loading {last_trial_log}")
            vis_log = torch.load(last_trial_log)
            logs[log_dir].append(vis_log)
            configs[log_dir].append(full_conf)

    for log_dir in logs_dirs:
        if os.path.exists(os.path.join(log_dir, 'config.yaml')):
            log.info(f"Loading latest trial from {log_dir}")
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
                if ld == '.slurm':
                    continue
                log_subdir = os.path.join(log_dir, ld)
                if os.path.isdir(log_subdir):
                    if file is None:
                        # Load data for the smallest trial number from all sub directories
                        if len(latest[log_dir]) == 0:
                            log.warn(f"No trial files found under {log_dir}")
                            break
                        trial_file = natsorted(latest[log_dir])[0]
                    else:
                        trial_file = file
                    load_log(log_subdir, trial_file)
    log.info(f"Loaded logs from {len(logs)} directories")
    vis.visualize_group(vis_obj, vis_env, logs, configs, filter_regex)


def visualize_sweep(vis_obj, sweep_dir, vis_env, filter_regex):
    dirs = []
    for ld in os.listdir(sweep_dir):
        d = os.path.join(sweep_dir, ld)
        if os.path.isdir(d) and ld != '_slurm_logs_':
            dirs.append(d)
    visualize_group(vis_obj, dirs, None, vis_env, filter_regex)


def visualize_dynamics(vis_obj, checkpoint_file, vis_env):
    log.info(f"Visualizing dynamics model in in {checkpoint_file}")
    logs = defaultdict(list)
    configs = defaultdict(list)

    checkpoint_dir = checkpoint_file[:checkpoint_file.rfind("/") + 1]
    # TODO generate the needed configs as done in visualize_group
    if os.path.exists(os.path.join(checkpoint_dir, 'config.yaml')):
        full_conf = OmegaConf.load(f"{checkpoint_dir}/config.yaml")
        log.info(f"Loading {checkpoint_file}")
        vis_log = torch.load(checkpoint_file)
        logs[checkpoint_file].append(vis_log)
        configs[checkpoint_file].append(full_conf)
    else:
        raise ValueError("Checkpoint directory missing expected config")

    vis.visualize_dynamics(vis_obj, vis_env, logs, configs)


log = logging.getLogger(__name__)


@hydra.main(config_path='config-vis.yaml')
def experiment(cfg):
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    vis_obj = visdom.Visdom(port=cfg.base.port)

    if cfg.func == 'visualize':
        sub_cfg = cfg.visualize
        visualize(vis_obj, sub_cfg.file, sub_cfg.vis_env, sub_cfg.delta, sub_cfg.sort)
    elif cfg.func == 'visualize_group':
        sub_cfg = cfg.visualize_group
        visualize_group(vis_obj, sub_cfg.logs, sub_cfg.file, sub_cfg.vis_env, filter_regex=None)
    elif cfg.func == 'visualize_sweep':
        sub_cfg = cfg.visualize_sweep
        visualize_sweep(vis_obj, sub_cfg.dir, sub_cfg.vis_env, sub_cfg.filter)
    elif cfg.func == 'visualize_dynamics':
        sub_cfg = cfg.visualize_sweep
        visualize_dynamics(vis_obj, sub_cfg.file, sub_cfg.vis_env)
    else:
        raise ValueError(f"Unknown command {cfg.func}")

    log.info(f"Ran Function {cfg.func}")
    log.info(f" sub-config {sub_cfg.pretty()}")


if __name__ == '__main__':
    sys.exit(experiment())
