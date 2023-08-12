#!/usr/bin/env python3
import glob
import os
import sys

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())

from time import time
from time import sleep as time_sleep
from datetime import datetime
import gym
import numpy as np
import torch
from mbrl.policies import RandomPolicy, SineWavesPolicy, CPGPolicy
from mbrl.dataset import SASDataset
from mbrl.dynamics_model import NNBasedDynamicsModel
# noinspection PyUnresolvedReferences
from mbrl import utils, environments
import hydra
import logging
import matplotlib.pyplot as plt

# TODO alonrot: Added
import _pickle
from tools.tools import create_new_folder,acquire_rewards,acquire_state,RAD2DEG,load_dynamics_model,load_data4analysis_and_training,create_file_name2save,normalize_episodes,unnormalize_episodes,get_mbrl_base_path
from plotting_and_analysis.plotting_library import plot_evolution_init,add_plot_evolution,save_plot_evolution
import pdb
import matplotlib.cm as colormap
import yaml

import mbrl.rewards


def save_log(cfg, trial_num, trial_log, planned_sequence):
    name = cfg.checkpoint_file.format(trial_num)
    path = os.path.join(os.getcwd(), name)
    log.info(f"T{trial_num} : Saving log {path}")
    torch.save(trial_log, path)
    if planned_sequence is not None:
        name = "planned_sequence_{}.dat".format(trial_num)
        path = os.path.join(os.getcwd(), name)
        log.info(f"T{trial_num} : Saving trajectories data to {path}")
        torch.save(planned_sequence, path)

def save_sas(cfg, trial_num, episode_SAS, planned_sequence, episode_raw_SAS, my_policy_params, path_base=None):

    if path_base is None:
        path_base = os.getcwd()
    else:
        assert isinstance(path_base,str)

    name = cfg.checkpoint_file.format(trial_num)
    path = os.path.join(path_base, name)
    log.info(f"T{trial_num} : Saving log {path}")

    base_name = cfg.motor_babbling.base_name2save

    name = "{0:s}_SAS_episode_{1:d}.dat".format(base_name,trial_num)
    path = os.path.join(path_base, name)
    log.info(f"T{trial_num} : Saving SAS trajectories to {path}")
    torch.save(episode_SAS, path)

    name = "{0:s}_raw_SAS_episode_{1:d}.dat".format(base_name,trial_num)
    path = os.path.join(path_base, name)
    log.info(f"T{trial_num} : Saving RAW SAS trajectories to {path}")
    torch.save(episode_raw_SAS, path)

    name = "{0:s}_planned_sequence_episode_{1:d}.dat".format(base_name,trial_num)
    path = os.path.join(path_base, name)
    log.info(f"T{trial_num} : Saving SAS trajectories to {path}")
    torch.save(planned_sequence, path)

    name = "{0:s}_SAS_arrays_episode_{1:d}.dat".format(base_name,trial_num)
    path = os.path.join(path_base, name)
    log.info(f"T{trial_num} : Saving SAS trajectories to {path}")

    name = "{0:s}_SAS_parameters_episode_{1:d}.dat".format(base_name,trial_num)
    path = os.path.join(path_base, name)
    log.info(f"T{trial_num} : Saving SAS parameters to {path}")

    fid = open(path,"wb")
    _pickle.dump(my_policy_params,fid)
    fid.close()

    return path_base

def save_episode(path_full,trial_num,ep):
    log.info("Saving Episode...")
    log.info("Saving using torch.save()...")
    log.info("trial_num: {0:d} | Saving to {1:s}".format(trial_num,path_full))
    torch.save(ep,path_full)

def save_reward(path_full,trial_num,reward):
    log.info("Saving Reward...")
    log.info("Saving using torch.save()...")
    log.info("trial_num: {0:d} | Saving to {1:s}".format(trial_num,path_full))
    torch.save(reward,path_full)

def split_and_append(episode, training_dataset, testing_dataset, cfg):

    log.info("Splitting the dataset using split: {0:1.1f}".format(cfg.training.testing.split))

    trial_dataset = SASDataset()
    trial_dataset.add_episode(episode, cfg.device)
    # import pdb;pdb.set_trace()
    training_subset, testing_subset = utils.split_dataset(trial_dataset, cfg.training)
    for sar in training_subset:
        training_dataset.add(sar)
    for sar in testing_subset:
        testing_dataset.add(sar)

def save_random_state(cfg,trial_num,is_init,path_base=None):

    if path_base is None:
        path_base = os.getcwd()
    else:
        assert isinstance(path_base,str)

    # Store here the random state. Actually, return it in the fucntion, the one before and rthe one after. This way, we shoudl be able to reproduce exactly whats' going on.
    state_random_np = np.random.get_state()
    state_random_torch = torch.random.get_rng_state()
    state_random_dict = dict(state_random_np=state_random_np,state_random_torch=state_random_torch)

    # pdb.set_trace()

    base_name = cfg.training.store_random_state.base_name2save
    if is_init:
        name = "{0:s}_SAS_episode_init.dat".format(base_name)
    else:
        name = "{0:s}_SAS_episode_{1:d}.dat".format(base_name,trial_num)
    path = os.path.join(path_base, name)
    log.info(f"T{trial_num} : Saving SAS Random State to {path}")
    torch.save(state_random_dict, path)

def split_and_append_with_shuffling(cfg,episode,normalize=False):

    """
    What eventually goes INTO the training, is training_dataset, testing_dataset, not episode. So
    We need do do something about this. Restart the datasets probably

    """

    assert cfg.training.shuffle_data == True # These options should always be tru, and hence, removed from daisy.yaml

    if normalize:
        log.warning("Data normalization is active")
        log.warning("Collected (real) data is never effectively altered (i.e., not normalized). We normalize it, copy it, and unnormalize it again.")
        log.info("Normalizing training dataset...")
        # pdb.set_trace() # episode[5].s0[0:5]
        # Normalize each state individually across all episodes:
        sas_mean, sas_std = normalize_episodes(episode)
        # pdb.set_trace()


    # Split data in training and testing datasets
    # See training.testing.split, in  config.yaml (by default 90%-10% split)
    log.info("Calling split_and_append: First shuffle all the data, then, split the datasets.")
    training_dataset = SASDataset()
    testing_dataset = SASDataset()
    split_and_append(episode, training_dataset, testing_dataset, cfg) # Classic way of splitting the dataset

    if normalize:
        log.info("UN-Normalizing training dataset...")
        # pdb.set_trace()
        unnormalize_episodes(episode,sas_mean,sas_std)
        # pdb.set_trace()

    return training_dataset, testing_dataset

def get_customized_dataset_shuffled(cfg,old_dataset,new_dataset,device,ratio_desired=0.05,shuffle_here=False):

    # Take a percentage of the old data set, and place into a reduced old data set

    # Keep a constant ratio between the reduced old data set and the new data, but replicating the new data.

    # For example. Old data set: 8000
    # Reduced old data set: 10% of that, i.e., 800 datapoints
    # New data set: 15 points.
    # Desired ratio between them: 20%, i.e., X/800 = 0.2 -> X -> 160
    # Replicate 15 to 160
    # 

    # assert isinstance(ratio_desired,float)
    # assert ratio_desired > 0. and ratio_desired < 1., "Desired ratio must be between 0 and 1"

    # # Needed length of new data:
    # Ndata_reduced = int(np.floor(len(old_dataset)*ratio_desired))
    # Ndata_rest = len(old_dataset) - Ndata_reduced

    # # Reduce old data set:
    # _,old_dataset_reduced = torch.utils.data.random_split(old_dataset, [Ndata_rest, Ndata_reduced])


    # Errorc checking:
    if len(new_dataset) == 0:
        pdb.set_trace()

    # Get length of new dataset:
    Ndata_old = len(old_dataset)
    Ndata_des = int(np.floor( Ndata_old*(ratio_desired/(1-ratio_desired)) ))
    Ndata_new = len(new_dataset)

    if Ndata_new >= Ndata_old:
        pdb.set_trace()

    # Replicate only when the new dataset is too small:
    log.info("Ratio r = (Ndata_new / (Ndata_new + Ndata_old)), with r = {0:f}".format(ratio_desired))
    new_dataset_extended = SASDataset()
    if Ndata_new < Ndata_des: # Replicate

        log.info("Episode data is being replicated. Acquired: {0:d} | Required: {1:d}".format(Ndata_new,Ndata_des))
        log.info("Ratio r is mantained")

        Nreplicas = int(np.ceil(Ndata_des / Ndata_new))

        # Replicate the data set Nreplicas-1 times, and later on add the remaining points:
        for k in range(Nreplicas-1):
            new_dataset_extended.add_episode(episode=new_dataset,device=device)

        rem_epi = Ndata_des - len(new_dataset_extended)
        for k in range(rem_epi):
            new_dataset_extended.add(new_dataset[k])

        # # Cut from the end the remaining samples: -> Didn't work: pop() is not part of Tensorlist, defined in dataset.py
        # for k in range(len(new_dataset_extended)-Ndata_des):
        #     new_dataset_extended.states0.pop(-1)
        #     new_dataset_extended.actions.pop(-1)
        #     new_dataset_extended.states1.pop(-1)

    else:
        log.info("Episode data is more than the required. Acquired: {0:d} | Required: {1:d}".format(Ndata_new,Ndata_des))
        log.info("Ratio r is not mantained")
        log.info("New ratio: Ndata_new / (Ndata_new + Ndata_old) = {0:f}".format(Ndata_new / (Ndata_new + Ndata_old)))
        new_dataset_extended.add_episode(episode=new_dataset,device=device)

    # Create a new data set that concatenates both, just for training.
    new_dataset_extended.add_episode(episode=old_dataset,device=device) # Extend it, to avoid creating an entire new one

    if shuffle_here:
        dataset4training = SASDataset()

        # Now, suffle the data:
        ind_samples = torch.utils.data.RandomSampler(new_dataset_extended, replacement=False)
        for ind in ind_samples:
            dataset4training.add(new_dataset_extended[ind])
    else:
        dataset4training = new_dataset_extended # This also new memory because new_dataset_extended is new memory

    # NOTE: dataset4training does NOT share memory with new_dataset_extended, nor old_dataset, as it is freshly created (!)

    return dataset4training


def find_latest_checkpoint(cfg):
    '''
    Try to find the latest checkpoint in the log directory if cfg.checkpoint
    is not provided (usually through the command line).
    '''
    # same path as in save_log method, but with {} replaced to wildcard *
    checkpoint_paths = os.path.join(os.getcwd(),
                                    cfg.checkpoint_file.replace("{}", "*"))

    # use glob to find files (returned a list)
    files = glob.glob(checkpoint_paths)

    # If we cannot find one (empty file list), then do nothing and return
    if not files:
        return None

    # find the one with maximum last modified time (getmtime). Don't sort
    last_modified_file = max(files, key=os.path.getmtime)

    return last_modified_file

def train_dynamics_model(cfg,dynamics_model,trial_num,training_dataset,testing_dataset,which_phase,path_walking_experiments):

    # ================
    # <<< Training >>>
    # ================

    file_name = "{0:s}_after_episode_{1:d}".format(cfg.walking_experiments.dynamics_model.name_base,trial_num)
    if which_phase == "init":
        cfg.training.epochs = cfg.training.full_epochs
        file_name = "{0:s}_init".format(cfg.walking_experiments.dynamics_model.name_base)
    elif which_phase == "episodes":
        cfg.training.epochs = cfg.training.incremental_epochs
    elif which_phase == "episode_wasnt_completed":
        cfg.training.epochs = cfg.training.incremental_epochs
    else:
        raise ValueError("which_phase = {'init','episodes','episode_wasnt_completed'}")

    log.info("======================================================")
    log.info("========== <<< Training at iter {0:2d} >>>> ==========".format(trial_num))
    log.info("======================================================")
    log.info("Training for {0:d} epochs...".format(cfg.training.epochs))
    log.info("Using {0:d} datapoints for training...".format(len(training_dataset)))
    time_init_train = datetime.utcnow().timestamp()
    train_log = dynamics_model.train(training_dataset, testing_dataset, cfg.training)
    train_log.trial_num = trial_num
    time_train_tot = datetime.utcnow().timestamp() - time_init_train

    # Save the model as it is retrained

    # Show some info in the screen:
    msg = f"T{trial_num} : Dynamics model trained"
    if 'train_loss' in train_log is not None:
        msg += f", train loss={train_log.train_loss:.4f}"
    if 'test_loss' in train_log is not None:
        msg += f", test loss={train_log.test_loss:.4f}"
    if 'total_time' in train_log and 'epochs' in train_log:
        eps = (train_log.epochs / train_log.total_time)
        msg += f" ({train_log.total_time:.2f}s, {train_log.epochs} epochs@{eps:.2f}/sec)"
    log.info(msg)

    log.info("Training finished!")

    # Gather some training info:
    training_info = dict(time_tot=time_train_tot,HOSTNAME=os.uname()[1],epochs=cfg.training.epochs,lr=cfg.training.optimizer.params.lr,batch_size=cfg.training.batch_size,cfg_device=cfg.device)

    # Save using torch.save():
    if cfg.walking_experiments.dynamics_model.save:
        log.info("Saving the trained dynamics model...")
        tuple2save = (dynamics_model,training_info)
        full_path = "{0:s}/{1:s}.{2:s}".format(path_walking_experiments,file_name,cfg.walking_experiments.dynamics_model.name_extension)
        torch.save(tuple2save,full_path)
    else: 
        log.info("NOT saving the trained dynamics model...")


    return dynamics_model


def init(cfg, env, path_base=None):
    if not cfg.checkpoint:
        cfg.checkpoint = find_latest_checkpoint(cfg)

    if not cfg.checkpoint: # TODO alonrot: normally, cfg.checkpoint = false
        if cfg.random_seed is not None:
            utils.random_seed(cfg.random_seed)
            env.seed(cfg.random_seed)
            torch.manual_seed(cfg.random_seed)

        # log.info(f"Dynamics model : {cfg.dynamics_model.clazz}")
        # dynamics_model = utils.instantiate(cfg.dynamics_model)
        # first_trial_num = 0

        # TODO alonrot: Training with a random policy:
        my_policy_params = []
        if cfg.env.data_collection_policy == "random":
            data_collection_policy = RandomPolicy(cfg.device, env.action_space, cfg.policy.params.planning_horizon)
        elif cfg.env.data_collection_policy == "sinewaves":
            data_collection_policy = SineWavesPolicy(cfg.device, env.action_space, cfg.policy.params.planning_horizon)
        elif cfg.env.data_collection_policy == "cpg":
            data_collection_policy = CPGPolicy(cfg.device, env.action_space, cfg.policy.params.planning_horizon,
                                                cfg.env.cpg_policy.offset_shoulder, cfg.env.cpg_policy.offset_elbow, cfg.env.action_size)
        else:
            raise ValueError("Incorrect cfg.env.data_collection_policy for data collection and NN training")

        if cfg.motor_babbling.checkpoint.use:
            assert cfg.motor_babbling.checkpoint.trial_init < cfg.motor_babbling.num_trials,    "motor_babbling.checkpoint.use = True in daisy.yaml; \
                                                                                                make sure this is what you want"
            trial_init = cfg.motor_babbling.checkpoint.trial_init
        else:
            trial_init = 0

        trial = trial_init
        episode_all = []
        while trial < cfg.motor_babbling.num_trials:

            # Execute actual episode in the real robot for data collection and training the dynamics model:
            ep, ep_raw, episode_completed = utils.sample_episode(   env, data_collection_policy, 
                                                                    cfg.motor_babbling.task_horizon, 
                                                                    cfg.env.freq_action, 
                                                                    return_raw_state=cfg.env.return_raw_state, 
                                                                    which_policy=cfg.env.data_collection_policy)
            episode_all += ep.episode
            
            go_on_with_the_loop = continue_or_repeat(episode_completed,ep)

            # We restart the episode after the user presses a key, and do NOT update the iteration number
            if not go_on_with_the_loop:
                log.info("Repeating the episode. All acquired data will be discarded.")
                # input("Press enter to continue: ")
                continue # We inmediately go back up to the beginning of the while loop, without updating trial_num (i.e., we repeat this iteration)

            if cfg.env.data_collection_policy == "cpg":

                # Resample parameters every:
                resample_tripod_every = cfg.env.cpg_policy.resample_tripod_every
                if isinstance(resample_tripod_every,int):
                    assert resample_tripod_every > 0
                elif resample_tripod_every is None or resample_tripod_every == "None":
                    resample_tripod_every = cfg.motor_babbling.num_trials + 1 # Never resample
                resample_tripod = (trial+1) % resample_tripod_every == 0
                logging.info("resample_tripod: {0:s}".format(str(resample_tripod)))

                data_collection_policy.reset_trajectory(resample_tripod=resample_tripod)
                my_policy_params = data_collection_policy.get_parameters()
                path = save_sas(cfg, trial, ep.episode, ep.planned_action_sequences, ep_raw.episode, my_policy_params,path_base=path_base)
            elif cfg.env.data_collection_policy == "sinewaves": # Re-sample the parameters, to have diversity

                # TODO alonrot: Added a line to save to file:
                my_policy_params = data_collection_policy.get_sinewaves_parameters()
                path = save_sas(cfg, trial, ep.episode, ep.planned_action_sequences, my_policy_params,path_base=path_base)

                if cfg.motor_babbling.resample_sinewaves_every is not None:
                    assert cfg.motor_babbling.resample_sinewaves_every > 0
                    assert isinstance(cfg.motor_babbling.resample_sinewaves_every,int)
                    if trial % cfg.motor_babbling.resample_sinewaves_every == 0:
                        logging.info("Resampling parameters... (resampling every {0:d} episodes. Total: {0:d} episodes)".format(cfg.motor_babbling.resample_sinewaves_every,cfg.motor_babbling.num_trials))
                        data_collection_policy.sample_sinewaves_parameters()
            else:
                raise NotImplementedError

            trial += 1

        log.info("Collected data in {0:d} episodes using policy {1:s}".format(len(ep.episode),cfg.env.data_collection_policy.upper()))

        # NOTE: The split_and_append can be done outside the for loop, because the training_dataset and testing_dataset are NOT saved/used within iterations.
    else:
        log.info(f"Loading checkpoint {cfg.checkpoint}")
        t = time()
        checkpoint = torch.load(cfg.checkpoint)
        utils.restore_rng_state(checkpoint, env)
        print(f"Loaded, took {(time() - t):.3f}s")
        assert checkpoint['env_name'] == cfg.env.name, \
            "Mismatch between loaded checkpoint and the environment you specified, {} != {}".format(
                checkpoint['env_name'], cfg.env.name)

        dynamics_model = checkpoint['dynamics_model']
        training_dataset = checkpoint['training_dataset']
        testing_dataset = checkpoint['testing_dataset']
        first_trial_num = checkpoint['trial_num'] + 1
        if isinstance(dynamics_model, NNBasedDynamicsModel):
            # When loading a checkpoint, be sure to respect the jit flag to allow debugging of model issues
            # dynamics_model.set_jit(cfg.dynamics_model.jit) # original
            dynamics_model.set_jit(cfg.dynamics_model.params.jit) # TODO alonrot added: Modified the above line. Shouldn't this be .params.jit, instead of .jit?

    return episode_all, path


log = logging.getLogger(__name__)
@hydra.main(config_path='conf/config.yaml', strict=True)
def collect_data(cfg):

    # log.info("Running on: {}".format(socket.gethostname()))
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    # Load environment:
    env = environments.daisy_real.DaisyRealRobotEnv(cfg.env)
    # env = gym.make(cfg.env.name) # Load simulators

    # Load reward function:
    # env.reward_func = utils.get_static_method(cfg.env.reward_func)
    env.reward_func = mbrl.rewards.RewardWalkForward(dim_state=env.get_state_dim())

    # Collect data:
    episode, path = init(cfg, env, path_base=None)

    print("Stopping robot...")
    env.stop_robot()

def continue_or_repeat(episode_completed,ep):

    # We restart the episode is the user wants to:
    if not episode_completed and len(ep.episode) == 0:
        log.info("Episode was never started as the robot reported bad status during env.reset() (i.e., while going to home position).")
        log.info("No data acquired. No data will be saved. The model won't be retrained. The iteration will be repeated")
        input("Press enter key to continue...")
        go_on_with_the_loop = False
    elif not episode_completed:
        log.info("Episode not completed due to robot bad status.")
        log.info("Do you want to repeat the episode [0] or continue [1] ?")
        log.info("If repeated: All the data will be discarded, and the model won't be retrained.")
        log.info("If continue: You will be able to select whether to retrain or not the model with the acquired data.")
        log.info("The data will be saved anyways.")
        ipt = 999
        while not ipt in ["0","1"]:
            ipt = input("Your choice: ")
        go_on_with_the_loop = ipt == "1"
    else:
        log.info("Episode was completed successfully!")
        log.info("This is a security prompt. In case something went wrong, which wasn't automatically detected, you can still discard the data and repeat the iteration")
        log.info("Do you want to repeat the episode [0] or continue [1] ?")
        log.info("If repeated: All the data will be discarded, and the model won't be retrained.")
        log.info("If continue: You will be able to select whether to retrain or not the model with the acquired data.")
        log.info("The data will be saved anyways.")
        ipt = 999
        while not ipt in ["0","1"]:
            ipt = input("Your choice: ")
        go_on_with_the_loop = ipt == "1"

    return go_on_with_the_loop

def plot_evolution_init_pets(cfg):

    indices2monitor = eval(cfg.walking_experiments.indices2monitor)
    assert isinstance(indices2monitor,list)
    assert len(indices2monitor) > 0
    indices2monitor = np.array(indices2monitor)
    rewards_list = [None]*cfg.num_trials
    state_list = [None]*cfg.num_trials

    hdl_fig, hdl_plot = plot_evolution_init(figsize=(10,8),indices2monitor=indices2monitor) # Soleve this by adding as many subplots as states
    colors = colormap.get_cmap("nipy_spectral")
    Ncolors = 6
    colors_mat = np.zeros((cfg.num_trials,3))
    for k in range(cfg.num_trials):
        # colors_mat[k,:] = colors((k % Ncolors)/Ncolors)[0:3] # Cycle over the spectrum every Ncolors steps
        colors_mat[k,:] = colors((k % cfg.num_trials)/cfg.num_trials)[0:3] # Cycle over the spectrum only once, ranging till cfg.num_trials

    return indices2monitor, rewards_list, state_list, colors_mat, hdl_fig, hdl_plot

def plot_evolution_add_plot_pets(cfg,ep,indices2monitor,trial_num,rewards_list,state_list,hdl_fig,hdl_plot,colors_mat,path_walking_experiments):

    # Keep track of rewards and relevant states evolution:
    rewards_list[trial_num] = acquire_rewards(ep)
    state_list[trial_num] = acquire_state(ep,indices2monitor)

    add_plot_evolution(hdl_plot,cfg,rewards_list[trial_num],state_list[trial_num],trial_num,indices2monitor,cfg.num_trials,colors_mat[trial_num,:],block=False) # Soleve this by adding as many subplots as states
    # plot_rewards(cfg,rewards_list[trial_num],state_list[trial_num],trial_num,path_walking_experiments)
    if cfg.walking_experiments.episodes.figure_evolution.save:
        save_plot_evolution(hdl_fig,path_walking_experiments,cfg.walking_experiments.episodes.name_base,trial_num)
    return rewards_list[trial_num]


def update_data4training(cfg,episodes_collected_all,ep_SAS_all):
    episodes_collected_all += ep_SAS_all
    log.info("Extending the global dataset from {0:d} to {1:d} datapoints. Added {2:d} points".format(len(episodes_collected_all)-len(ep_SAS_all),len(episodes_collected_all),len(ep_SAS_all)))
    return episodes_collected_all

    # # pdb.set_trace()
    # assert cfg.training.retrainNN_with == "cumulated_episode_data"

    # # Put data in the global list:
    # if cfg.training.retrainNN_with == "cumulated_episode_data":
    # elif cfg.training.retrainNN_with == "single_episode_data":
    #     episodes_collected_all = ep_SAS_all
    # else:
    #     raise ValueError("cfg.training.retrainNN_with: {cumulated_episode_data,single_episode_data}")


def shorten_and_shuffle(cfg,dataset):

    per_red = cfg.training.shorten_initial_dataset.reduce_to
    assert isinstance(per_red,int) or isinstance(per_red,float)
    assert per_red > 0. and per_red < 1.0

    Ndataset_short = int(np.ceil(per_red*len(dataset)))

    # Now, suffle the data:
    dataset_short = SASDataset()
    ind_samples = torch.utils.data.RandomSampler(dataset, replacement=False)
    k = 0
    for ind in ind_samples:
        if k == Ndataset_short:
            break
        dataset_short.add(dataset[ind])
        k += 1

    assert len(dataset_short) == Ndataset_short
    return dataset_short


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
@hydra.main(config_path='conf/config.yaml', strict=True)
def experiment(cfg):
    """
    Analyze pre-stored trained model by running PETS episodes on the real robot (WITH retraining the model with the new observed data)
    """

    # Seeds:
    # utils.random_seed(cfg.random_seed)
    # env.seed(cfg.random_seed)
    if cfg.random_seed is not None:
        torch.manual_seed(torch.randint(cfg.random_seed*100,(1,1)))

    # log.info("Running on: {}".format(socket.gethostname()))
    log.info("============= Configuration =============")
    log.info(f"Config:\n{cfg.pretty()}")
    log.info("=========================================")

    # Special options if running a profiler:
    if cfg.use_profiler:
        cfg.env.communication.time2pos_reset = 0.1
        cfg.env.communication.time2sleep_after_reset = 0.1
        cfg.env.communication.wait4stabilization = False
        cfg.num_trials = 10
        cfg.trial_timesteps = 300
        cfg.training.do_train_over_episodes = False

        # cfg.env.communication.HOST = "localhost"
        # cfg.env.communication.UDP_IP = "localhost"

    # Load environment:
    env = environments.daisy_real.DaisyRealRobotEnv(cfg.env)
    # env = gym.make(cfg.env.name) # Load simulators

    # Device:
    device_curr = cfg.device
    log.info("device_curr: {0:s}".format(device_curr))

    # Load reward function:
    log.info(f"Reward function : {cfg.reward.clazz}")
    reward_class = utils.instantiate(cfg.reward)
    env.setup_reward_function(reward_class.get_reward_signal)

    # # Initialize empty struct:
    # training_dataset = SASDataset()
    # testing_dataset = SASDataset()

    # Keep a separate structure to keep track of all the episodes, renormalized altogether at each iteration:
    episodes_collected_all = []

    # Load and parse:
    if cfg.initialization_type == "load_model":

        data_and_opts = load_data4analysis_and_training(cfg)
        path = data_and_opts["path"]
        folder2model = data_and_opts["folder2model"]
        model_name = data_and_opts["model_name"]
        ep_SAS_all = data_and_opts["ep_SAS_all"] # This is a list of episodes

        # Put data in the global list depending on what the user wants. See the field training.retrainNN_with: {"cumulated_episode_data","single_episode_data"}
        episodes_collected_all = update_data4training(cfg,episodes_collected_all,ep_SAS_all)
        # pdb.set_trace()

        # # Load datasets:
        # training_dataset, testing_dataset = split_and_append_with_shuffling(cfg,episodes_collected_all,normalize=cfg.data_normalization.use) 

        # Load the trained model:
        dynamics_model = load_dynamics_model(path,folder2model,model_name,device_curr)
    
        # Get the path to the walking experimetns:
        path_walking_experiments = "{0:s}/{1:s}/".format(path,cfg.walking_experiments.folder2save)
        path_walking_experiments,_ = create_new_folder(path_walking_experiments,append_timeofday=True,ask_user=False)
        
    elif cfg.initialization_type == "collect_data":

        # Load dynamics model:
        log.info(f"Dynamics model : {cfg.dynamics_model.clazz}")
        dynamics_model = utils.instantiate(cfg.dynamics_model)

        # If we are on a checkpoint, we need to know in which path. If not, a new path will be created
        if cfg.motor_babbling.checkpoint.use:
            data_and_opts = load_data4analysis_and_training(cfg)
            path_base = data_and_opts["path"]
            del data_and_opts

            all_files = os.listdir(path_base)
            if len(all_files) > 0 or len(all_files) == 1 and not 'main.log' in all_files:
                log.warning("Directory is not empty (!)\nMake sure the specified cfg.motor_babbling.checkpoint.trial_init is the one you want")
                log.warning("Directory: {0:s}".format(path_base))
                input("If correct, press enter to continue: ")

        else:
            path_base = None

        # Collect data:
        episode, path = init(cfg, env, path_base=path_base)

        # Put data in the global list depending on what the user wants. See the field training.retrainNN_with: {"cumulated_episode_data","single_episode_data"}
        episodes_collected_all = update_data4training(cfg,episodes_collected_all,episode)

        save_random_state(cfg,trial_num,is_init=True,path_base=path_base)

        # Load datasets:
        training_dataset, testing_dataset = split_and_append_with_shuffling(cfg,episodes_collected_all,normalize=cfg.data_normalization.use)

        # Get the path to the walking experimetns:
        path_walking_experiments = "{0:s}/{1:s}/".format(path,cfg.walking_experiments.folder2save)
        path_walking_experiments,_ = create_new_folder(path_walking_experiments,append_timeofday=True,ask_user=False)

        # Train the model:
        dynamics_model = train_dynamics_model(  cfg=cfg,dynamics_model=dynamics_model,trial_num=-1,
                                                training_dataset=training_dataset,testing_dataset=testing_dataset,
                                                which_phase="init",path_walking_experiments=path_walking_experiments)

    else:
        raise ValueError("initialization_type = {'load_model','collect_data'}")

    # We want to have something like this:

    """
    ./output/day/time/ -> typially created root folder for data collection. Store here logs, and ep.episode, coming out of init().
    ./output/day/time/cluster_data/ -> Store here the offline trained dynamics model.
    ./output/day/time/walking_experiments/day/time/ -> Store here the dynamics model trained during PETS, as a checkpoint.
    ./output/day/time/walking_experiments/day/time/ -> Store also the episodes data and the model analysis data.
    ./output/day/time/walking_experiments/day/time/
    NOTE: if using save_sas() to save both, the init() data and the normal PETS episodes data, make sure passing a name_base, otherwise data will be replaced...
    NOTE: Have the option of starting from a folder that only contains init data, but no offline trained model yet. Add the possibility to
            redo the init() from a non-zero iteration number, and add data to the existing one. Then, retrain the model.
    """

    # Shorten training dataset, when it is too large:
    if cfg.training.shorten_initial_dataset.use:
        episodes_collected_all = shorten_and_shuffle(cfg,episodes_collected_all)
        log.info("Initial dataset reduced by a {0:f} % (!)".format(cfg.training.shorten_initial_dataset.reduce_to*100))

    # Load policy:
    policy = utils.instantiate(cfg.policy, cfg)
    if repr(policy) == 'PETSPolicyParametrized':
        # policy.setup(dynamics_model, env.parameter_space, utils.get_static_method(cfg.env.reward_func))
        policy.setup(dynamics_model, env.parameter_space, env.reward_func)
    else:
        # policy.setup(dynamics_model, env.action_space, utils.get_static_method(cfg.env.reward_func))
        policy.setup(dynamics_model, env.action_space, env.reward_func)

    if cfg.walking_experiments.episodes.figure_evolution.plot:
        indices2monitor, rewards_list, state_list, colors_mat, hdl_fig, hdl_plot = plot_evolution_init_pets(cfg)

   

    if cfg.training.perform_trainning_every_xxx_datapoints.use and cfg.training.perform_trainning_every_xxx_episodes.use:
        raise ValueError("Decide either perform_trainning_every_xxx_datapoints or perform_trainning_every_xxx_episodes")

    if cfg.training.perform_trainning_every_xxx_episodes.use:
        assert cfg.training.perform_trainning_every_xxx_episodes.xxx < cfg.num_trials, "You want to train every {0:d} episodes, but the max. number of episodes is {1:d}".format(cfg.training.perform_trainning_every_xxx_episodes.xxx,cfg.num_trials)

    if cfg.training.perform_trainning_every_xxx_datapoints.use:
        assert cfg.training.perform_trainning_every_xxx_datapoints.xxx < cfg.num_trials*cfg.trial_timesteps


    # ======================
    # <<< Main PETS loop >>>
    # ======================
    trial_num = 0
    # new_data_acquired = True # This will always be true, except for a few corner cases
    episodes_effective = 0
    epoisodes4training = []
    while trial_num < cfg.num_trials:

        log.info("Reset the input trajectory to the optimizer. As a non-informative choice, we start with a static pose along the planning horizon. Such static pose is just standing up position (which coincides with the middle of the interval")
        policy.reset_trajectory()

        # TODO alonrot: Sample episode
        # TODO: Set a timeout for status_robot to 5 seconds or so before the episode, and to Inf after the episode.
        log.info("Starting a new episode...")
        # env.set_status_robot_timeout(10.)
        ep, ep_raw, episode_completed = utils.sample_episode(env, policy, cfg.trial_timesteps, cfg.env.freq_action, 
                                                            return_raw_state=True, render=cfg.env.render, which_policy=None)

        # Repeat episode and discard the data or continue and use/not use the data:
        go_on_with_the_loop = continue_or_repeat(episode_completed,ep)
        if not go_on_with_the_loop:
            log.info("Repeating the episode. All acquired data will be discarded. No data will be saved. The model won't be retrained.")
            continue # We inmediately go back up to the beginning of the while loop, without updating trial_num (i.e., we repeat this iteration)        
        
        # Plot evolution:
        if cfg.walking_experiments.episodes.figure_evolution.plot:
            rewards_curr = plot_evolution_add_plot_pets(cfg,ep,indices2monitor,trial_num,rewards_list,state_list,hdl_fig,hdl_plot,colors_mat,path_walking_experiments)

        # Have here a proper saving function for the love of god
        if cfg.walking_experiments.episodes.save:
            path_full = "{0:s}/{1:s}_{2:d}.{3:s}".format(path_walking_experiments,cfg.walking_experiments.episodes.name_base,trial_num,cfg.walking_experiments.episodes.name_extension)
            save_episode(path_full,trial_num,ep)

        # Have here a proper saving function for the love of god
        if cfg.walking_experiments.rewards.save:
            if cfg.walking_experiments.episodes.figure_evolution.plot:
                path_full = "{0:s}/{1:s}_{2:d}.{3:s}".format(path_walking_experiments,cfg.walking_experiments.rewards.name_base,trial_num,cfg.walking_experiments.rewards.name_extension)
                save_reward(path_full,trial_num,rewards_curr)
            else:
                log.warning("Rewards not being saved (!) User wants to save them, but plotting is not activated...")

        # After (maybe) seeing the recorded data, we decide whether we want to use it to retrain the model or not.
        if cfg.training.do_train_over_episodes:
            if not episode_completed:
                log.info("Do you want to use [1] or not use [0] the recorded chunk of data to retrain the model?")
                log.info("The data has been saved anyways")
                ipt = 999
                while not ipt in ["0","1"]:
                    ipt = input("Your choice: ")
                new_data_acquired = ipt == "1"
                which_phase = "episode_wasnt_completed"
            else:
                new_data_acquired = True
                which_phase = "episodes"

            if new_data_acquired:

                # If the episode wasn't discarded, and it's suitable for retraining the model, we count it as 'effective':
                episodes_effective += 1
                epoisodes4training += ep.episode
                log.info("Append episode to the local list of episodes, filled while not training...")
                log.info("len(epoisodes4training): {0:d}, with episodes_effective: {1:d}".format(len(epoisodes4training),episodes_effective))

                if cfg.training.perform_trainning_every_xxx_episodes.use and episodes_effective % cfg.training.perform_trainning_every_xxx_episodes.xxx != 0:
                    log.warning("NO training. We only do training every {0:d} episodes. Current iteration index: {1:d}".format(cfg.training.perform_trainning_every_xxx_episodes.xxx,trial_num))
                elif cfg.training.perform_trainning_every_xxx_datapoints.use and len(epoisodes4training) < cfg.training.perform_trainning_every_xxx_datapoints.xxx:
                    log.warning("NO training. We only do training every {0:d} datapoints. Cumulated datapoints:: {1:d}".format(cfg.training.perform_trainning_every_xxx_datapoints.xxx,len(epoisodes4training)))
                else:
                
                    save_random_state(cfg,trial_num,is_init=False,path_base=path_walking_experiments)

                    # Get customized data set that includes data replication if the aquired episode is less than the required ratio_desired (normally a 10%):
                    dataset4training = get_customized_dataset_shuffled(cfg=cfg,old_dataset=episodes_collected_all,new_dataset=epoisodes4training,device=device_curr,ratio_desired=cfg.training.ratio_desired,shuffle_here=False)
                    # pdb.set_trace()

                    training_dataset, testing_dataset = split_and_append_with_shuffling(cfg,dataset4training,normalize=cfg.data_normalization.use)

                    dynamics_model = train_dynamics_model(cfg=cfg,dynamics_model=dynamics_model,trial_num=trial_num,training_dataset=training_dataset,testing_dataset=testing_dataset,
                                        which_phase=which_phase,path_walking_experiments=path_walking_experiments)

                    log.info("Resetting the local list of episodes used for training...")
                    epoisodes4training = []

                # Put data in the global list depending on what the user wants. See the field training.retrainNN_with: {"cumulated_episode_data","single_episode_data"}
                log.info("Append the collected episode to the global data set. This data is appended RAW. No suffling. No normalization.")
                episodes_collected_all = update_data4training(cfg,episodes_collected_all,ep.episode)

        else:
            log.warning("Training is NOT happening, as requested by the user (!)")

        # Security prompt:
        log.info("We have tried so far {0:d} episodes, among which {1:d} were used for training".format(trial_num+1,episodes_effective))
        log.info("This was iteration {0:d} (zero-based index)".format(trial_num))
        ipt = 999
        while not ipt in ["1"]:
            ipt = input("Enter 1 to continue to the next episode (the pets/conf/dynamic_conf.yaml file will be read): ")

        path_mbrl = get_mbrl_base_path()
        path2dynamic_conf = "{0:s}/{1:s}".format(path_mbrl,"pets/conf/dynamic_conf.yaml")
        fid = open(path2dynamic_conf,'r')
        aaa = yaml.load(fid)
        if aaa['activate'] == True:
            log.warning("User promt open to dynamically change configuration parameters. You are encouraged to change the contents of cfg, but nothing else! Use it with care...")
            log.warning("Once the changes have been made, type c and press enter")
            log.warning("For this message never to show up again, change the value 'activate' in the file {0:s}".format(path2dynamic_conf))
            pdb.set_trace()
        else:
            log.info("No dynamic configuration requested")


        # Counter update:
        trial_num += 1


    # TODO alonrot: Stop the robot
    log.info("Stopping robot...")
    env.stop_robot()
    log.info("Done!!")

    if cfg.walking_experiments.episodes.figure_evolution.plot:
        plt.show(block=cfg.walking_experiments.episodes.figure_evolution.block_at_end)

if __name__ == '__main__':
    
    # PETS main: Flexible version of the main to handle real robot experiment:
    sys.exit(experiment())

    # Only data collection:
    # sys.exit(collect_data())


    