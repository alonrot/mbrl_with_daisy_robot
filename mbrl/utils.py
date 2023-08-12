import hashlib
import logging
import math
import os

import numpy as np
import torch

from mbrl.dataset import Dataset
from mbrl.dataset import SAS
import inspect
log = logging.getLogger(__name__)

# alonrot:
from datetime import datetime,timedelta
import time
import copy
from mbrl.dataset import SASDataset
import pdb
# from mbrl.optimizers.optimizer import ActionSequence


def fullname(o):
    if inspect.isclass(o):
        return o.__module__ + "." + o.__qualname__
    else:
        return o.__module__ + "." + o.__class__.__qualname__


def get_class(path):
    try:
        from importlib import import_module
        module_path, _, class_name = path.rpartition('.')
        mod = import_module(module_path)
        klass = getattr(mod, class_name)
        return klass
    except ValueError as e:
        print("Error initializing class " + path)
        raise e


def get_static_method(full_method_name):
    try:
        spl = full_method_name.split('.')
        method_name = spl.pop()
        class_name = '.'.join(spl)
        clz = get_class(class_name)
        return getattr(clz, method_name)
    except Exception as e:
        log.error(f"Error getting static method {full_method_name} : {e}")
        raise e


def instantiate(config, *args):
    try:
        clazz = get_class(config.clazz)
        return clazz(*args, **({} if 'params' not in config else config.params))
    except Exception as e:
        log.error(f"Error instantiating {config.clazz} : {e}")
        raise e


def readable_dir(d):
    if not os.path.isdir(d):
        raise Exception("readable_dir:{0} is not a valid path".format(d))
    if os.access(d, os.R_OK):
        return d
    else:
        raise Exception("readable_dir:{0} is not a readable dir".format(d))


def readable_file(file):
    if not os.path.isfile(file):
        raise Exception("readable_file:{0} is not a valid path".format(file))
    if os.access(file, os.R_OK):
        return file
    else:
        raise Exception("readable_file:{0} is not a readable file".format(file))


def boolean_string(s):
    s = str.lower(s)
    if s not in {'false', 'true', 'yes', 'no'}:
        raise ValueError('Not a valid boolean string')
    return s in {'true', 'yes'}


class Episode:
    def __init__(self):
        self.episode = []
        self.rewards = []
        self.planned_action_sequences = []

def my_sleep(dt):

    startTime = datetime.now()
    sleepTime = timedelta(seconds = dt)

    while sleepTime > datetime.now() - startTime:
        pass

def sample_episode(env, controller, task_horizon, freq_action, return_raw_state=False, render=False, which_policy=None):
    ep = Episode()
    ep_raw = Episode()
    episode_completed = True

    # Required time steps:
    # import pdb;pdb.set_trace()
    dt = 1./freq_action
    Nsteps = int(task_horizon)
    print("Experiment parameters:\n    Commands sent at freq = "+str(freq_action)+", for t ~= "+str(Nsteps*dt)+" sec.")
    print("    Required Nsteps = "+str(Nsteps))
    print("    Action dt = "+str(dt))
    # input('    Press any key to start...')

    # Time buffer to send actions:
    t_action_buffer = 0.004

    # Some defs:
    action_numpy = np.zeros(18)
    action = torch.Tensor(size=(18,))

    if render:
        env.render()

    # TODO alonrot: env.reset() returns the state of the robot after resetting the environment, i.e., the initial state
    # For the real robot, we'll need to see what's involved in the resetting first, and then decide on what initial state we 
    # select
    # import pdb;pdb.set_trace()

    # Reset the robot by driving it to the initial position:
    state_from_reset_numpy_arr, is_alive = env.reset()
    if not is_alive:
        episode_completed = False
        if return_raw_state:
            return ep, ep_raw, episode_completed
        else:
            return ep, episode_completed


    # Set the state offsets to the last state observed after resetting.
    # Calling this function assumes that the robot will start moving right after.
    env.update_offsets(state_from_reset_numpy_arr) # Passing Raw state
    env.collect_observations_and_update(state_from_reset_numpy_arr)

    # Convert to torch:
    # state = torch.from_numpy(env.get_state_transformed()).to(torch.float32)
    state = torch.from_numpy(env.get_state_transformed()).to(device=env.device,dtype=torch.float32) # If not passing dtype=torch.float32, it will by default inherit the numpy dtype, i.e., float64, and will cause problems later

    for t in range(0, task_horizon):

        t_init = datetime.utcnow().timestamp()

        # TODO alonrot: We re-plan from a new state at each iteration:
        # import pdb; pdb.set_trace()
        # print("repr(controller):",repr(controller))
        if repr(controller) == 'PETSPolicyParametrized': # controller is not a string, it's a class, with the __repr__() method reimplemented
            action_sequence = controller.plan_action_sequence(state,t*dt)
        else:
            action_sequence = controller.plan_action_sequence(state)

        # We need a deep copy, otherwise, it's a list of pointer, which all point to the same memory address, i.e., all elements of the list are the same (the last return)
        # TODO: This can be fixed by pre-storing a list of action sequences outside this loop, and filling itw
        # ep.planned_action_sequences.append(copy.deepcopy(action_sequence)) # This deepcopy takes ~0.6 ms on average, as a result of a test (the test was done in-code, and erased)
        ep.planned_action_sequences.append(action_sequence) # This deepcopy takes ~0.6 ms on average, as a result of a test (the test was done in-code, and erased)

        # import pdb; pdb.set_trace()

        t_end_pets = datetime.utcnow().timestamp()
        print("\nEPISODE: t_iter="+str(t+1)+" / "+str(task_horizon)+" (steps); PETS planning overhead, before running env.step(): "+str((t_end_pets - t_init)*1000) + " [ms]")

        print("Sleeping for the rest of the desired time step, minus the buffer for env.step()")
        t_to_reach_action_buffer = t_init + dt - t_end_pets - t_action_buffer
        if t_to_reach_action_buffer < 0.0:
            print("[WARNING:] THE LOOP IS SATURATED (1) | t_iter="+str(t+1)+"; t_end_pets-t_init = "+str((t_end_pets-t_init)*1000)+" [ms]; dt = "+str(dt*1000) + " [ms]")
            print("t_to_reach_action_buffer < 0.0")
        else:
            my_sleep(t_to_reach_action_buffer)
            # time.sleep(t_to_reach_action_buffer)

        # TODO alonrot: Take the action at time t, on the replanned action sequence
        # TODO          get_action(state, t): state and t are IGNORED, is this the intention?
        # TODO          get_action(state, t) = self.actions[0].clone()
        if which_policy == "sinewaves":
            # print("Here!!!")
            t_offset = dt - t_action_buffer
            time2wave = t*dt + t_offset
            print("time2wave: {0:2.2f}".format(time2wave))
            sine_wave_vec = action_sequence.get_action(state, time2wave)
            action_numpy[:] = state_from_reset_numpy_arr[0:18] + sine_wave_vec # TODO: t*dt is not completely accurate here, as the action will take effect later on, i.e., at time t*dt + t_to_reach_action_buffer
            # action = torch.from_numpy(action_numpy).to(torch.float32) # Doint .to(torch.float32) is a bad idea, as it puts it by default in CPU (I think, test in robodev)
            action = torch.from_numpy(action_numpy).to(device=env.device,dtype=torch.float32)
            # action[...] = torch.from_numpy(action_numpy).to(torch.float32) # This makes all the ep.episode[0].a, ep.episode[1].a, ep.episode[2].a, ... to be all the SAME
            # print("action:\n",action)
            # print("action_numpy:\n",action_numpy)
            # print("sine_wave_vec:\n",sine_wave_vec)
            # print("state_from_reset_numpy_arr[0:18]:\n",state_from_reset_numpy_arr[0:18])
            # print("There is something wrong with the action that gets appended. For some reason, it is all saved as the same vector")
            
            # Suggestion, replace the above line
            # action[...] = torch.from_numpy(action_numpy).to(torch.float32)
            # by
            # action = torch.from_numpy(action_numpy).to(torch.float32)

            # raise

        elif which_policy == "cpg":
            action = action_sequence.get_action(state, t)
        else:
            action = action_sequence.get_action(state, t*dt)    # Is passing t here correct? -> No. We need to pass t*dt
                                                                # To be coherent with the function ActionSequenceSines.generate_action_sequence(),
                                                                # we need t_vec[0], where t_vec = t_curr + np.arange(0,planning_horizon)*dt, 
                                                                # which is actually t_vec[0] = t_curr. So, t_curr will do it, with t_curr = t*dt
            # When using the standard PETS policy (just trajectories), get_action() returns return self.actions[0].clone(), i.e., it ignores both state and t

            # TODO alonrot: Remove this (debug)
            if repr(action_sequence) == "ActionSequenceSines":
                action_debug = action_sequence.get_first_action_from_last_generated_sequence() # Equivalent to the above for the SineWavesPolicy

        if not torch.is_tensor(action):
            # action = torch.from_numpy(action).to(device=controller.optimizer.device)
            action = torch.from_numpy(action).to(device=controller.device,dtype=torch.float32)

        assert action.dim() == 1
        assert action.shape[0] == 18

        # Jump to a new state using the current action
        new_state, reward, is_alive = env.step(action.cpu().numpy())

        if not is_alive:
            episode_completed = False
            break

        assert len(new_state) == env.get_state_dim() # Comparing transformed state
        print("new_state (numpy):")
        print(new_state)
        new_state = torch.from_numpy(new_state).to(device=env.device,dtype=torch.float32) # TODO: The created tensor can be prestored

        # import pdb;pdb.set_trace()

        # TODO alonrot: Collect data
        if torch.is_tensor(reward):
            assert reward.dim() == 1
        else:
            assert reward.ndim == 1
        assert reward.shape[0] == 1
        print("reward: {0:2.2f}".format(reward[0]))
        
        # print("new_state (tensor):",new_state)
        ep.rewards.append(reward)

        if render:
            env.render()

        # TODO alonrot: This allocates memory dynamically, and can be avoided:
        ep.episode.append(SAS(state, action, new_state))
        ep_raw.episode.append(SAS(env.get_state_raw(return_tensor=True), action, env.get_state_raw(return_tensor=True)))

        # import pdb; pdb.set_trace()

        # TODO alonrot: Update state
        state = new_state

        # Take local loop time:
        t_end_action = datetime.utcnow().timestamp()
        t_elapsed_action = t_end_action - (t_init + dt - t_action_buffer)

        print("EPISODE: t_iter="+str(t+1)+" / "+str(task_horizon)+" (steps); PETS+action "+str((t_end_action - t_init)*1000.) + " [ms]")
        print("EPISODE: t_send_action="+str((t_elapsed_action)*1000) + " [ms]")
        print("EPISODE: t_action_buffer="+str(t_action_buffer*1000) + "[ms]")

        # Sleep once more, unless the loop is saturated:
        t_loop = t_end_action - t_init
        if t_end_action - t_init >= dt:
            print("[WARNING:] THE LOOP IS SATURATED (2) | t_iter="+str(t+1)+"; t_heavy_loop = "+str(t_loop*1000)+" [ms]; dt = "+str(dt*1000) + " [ms]")
        else:
            my_sleep(t_action_buffer-t_elapsed_action)
            # time.sleep(t_action_buffer-t_elapsed_action)

        # Measure the loop time once more:
        t_end_loop = datetime.utcnow().timestamp()

        print("EPISODE: t_iter="+str(t+1)+" / "+str(task_horizon)+" (steps); Loop time = "+str((t_end_loop - t_init)*1000) + " [ms]")

    # return ep, ep_raw
    if return_raw_state:
        return ep, ep_raw, episode_completed
    else:
        return ep, episode_completed


def moment_matching(means, variances):
    """
    Computes the uniformly weighted mean and variance for the input Gaussian distributions
    :param means: a list of numpy arrays representing the means
    :param variances: a list of numpy arrays representing the variances
    :return: mean and variance of the input lists (Moment matching, see https://arxiv.org/pdf/1612.01474.pdf,
             bottom part of '2.4 Ensembles: training and prediction'
    """
    means = np.array(means)
    variances = np.array(variances)
    assert means.shape == variances.shape
    # https://arxiv.org/pdf/1612.01474.pdf
    # Bottom part of '2.4 Ensembles: training and prediction'
    mean = means.sum(axis=0) / means.shape[0]
    variance = ((variances + means * means).sum(axis=0) / means.shape[0]) - (mean * mean)
    # due to numerical instability, the above formula may incorrectly sometime return very small negative values
    # variance should always be non negative, to avoid taking sqrt of negative values later I min clamp it by 0.
    # from torch.sqrt() let's min_clamp it with 0.
    variance = np.clip(variance, 0, 1e9)
    return mean, variance


def moment_matching_torch(means, variances):
    """
    Computes the uniformly weighted mean and variance for the input Gaussian distributions
    :param means: a list of torch tensor with the means
    :param variances: a list of torch tensor with the variances
    :return: mean and variance of the input tensors (Moment matching, see https://arxiv.org/pdf/1612.01474.pdf,
             bottom part of '2.4 Ensembles: training and prediction'
    """
    assert isinstance(means, list) or torch.is_tensor(means)
    assert isinstance(variances, list) or torch.is_tensor(variances)
    if isinstance(means, list):
        means = torch.stack(means)
    if isinstance(variances, list):
        variances = torch.stack(variances)

    # https://arxiv.org/pdf/1612.01474.pdf
    # Bottom part of '2.4 Ensembles: training and prediction'
    mean = means.mean(dim=2)
    variance = ((variances + means * means).sum(dim=2) / means.size(2)) - (mean * mean)
    # due to numerical instability, the above formula may incorrectly sometime return very small negative values
    # variance should always be non negative, to avoid taking sqrt of negative values later I min clamp it by 0.
    # from torch.sqrt() let's min_clamp it with 0.
    variance = torch.clamp_min(variance, 0)
    return mean, variance


# inplace truncated normal function for pytorch.
# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
# and tested to be equivalent to scipy.stats.truncnorm.rvs
def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def convert_to_dataset(sasdataset, target_transformer, state_transformer):
    """
    Converts a SASDataset (of (State, Action) -> State) into a standard input -> output dataset
    Input states are being transformed using the state transformer, and target states are
    potentially converted to deltas
    finally the actions are appended to the states, to form the inputs

    :param sasdataset:
    :param target_transformer: function(s0, s1) that returns the desired s1, typically a delta from into a delta from s0
    :param state_transformer: function(s) that converts input state into more nn-friendly input form
    :return:
    """
    x = []
    y = []
    for value in sasdataset:
        s0 = value.s0
        action = value.a
        s1 = value.s1

        # converts s1 into a delta from s0, this makes learning easier.
        # note that this need to be undone (delta state should be converted back to absolute state
        # when computing trajectories
        s1 = target_transformer(s0, s1)

        input1 = torch.cat([state_transformer(s0), action])

        x.append(input1)
        y.append(s1)

    return Dataset(torch.stack(x), torch.stack(y))


def split_to_subsets(dataset, num_splits):
    if num_splits == 1:
        # for 1 split, do not re-shuffle dataset to make it easier to compare using the dataset directly on a D or P
        # model to using PE or DE with an ensemble size of 1.
        return [dataset]
    ds = dataset
    chunk_sz = math.ceil(len(dataset) / num_splits)
    splits = []
    while len(splits) < num_splits:
        remainder = len(ds)
        if remainder <= chunk_sz:
            splits.append(ds)
        else:
            s1, ds = torch.utils.data.random_split(ds, [chunk_sz, remainder - chunk_sz])
            splits.append(s1)

    datasets = [[] for i in range(num_splits)]
    for i in range(num_splits):
        for j, ds in enumerate(splits):
            if i != j:
                datasets[i].append(ds)

    res = [torch.utils.data.ConcatDataset(datasets[k]) for k in range(num_splits)]
    return res


def md5sum(value):
    m = hashlib.md5()
    m.update(value.encode('utf-8'))
    return m.hexdigest()


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def assert_no_nans(py):
    if isinstance(py, dict):
        for v in py.values():
            assert not torch.isnan(v).any(), "nan detected in network output"
    else:
        assert not torch.isnan(py).any(), "nan detected in network output"


def split_dataset(dataset, training_params):
    training_dataset = dataset
    testing_dataset = None
    total_len = len(dataset)
    assert total_len > 1
    if training_params.testing is not None \
            and training_params.testing.split is not None \
            and training_params.testing.split != 1:
        train_len = math.floor(total_len * training_params.testing.split)
        test_len = total_len - train_len
        training_dataset, testing_dataset = torch.utils.data.random_split(dataset, [train_len, test_len])
    elif training_params.testing.split == 1: # Leave a dataset with only one element, for compatibility
        training_dataset, testing_dataset = torch.utils.data.random_split(dataset, [total_len-1, 1])
    else:
        raise ValueError

    return training_dataset, testing_dataset

def split_first_shuffle_after(dataset,len_training_dataset,device,shuffle=True):
    """
    author: alonrot
    """

    Ntot = len(dataset)
    if isinstance(len_training_dataset,int):
        assert len_training_dataset <= Ntot and len_training_dataset >= 0, "The passed index is incorrect"
    elif isinstance(len_training_dataset,float) and len_training_dataset > 0 and len_training_dataset < 1:
        len_training_dataset = int(np.ceil(Ntot*len_training_dataset))
    else:
        raise ValueError

    training_dataset = SASDataset()
    testing_dataset = SASDataset()

    if len_training_dataset == Ntot:
        training_dataset.add_episode(dataset,device)
    elif len_training_dataset == 0:
        testing_dataset.add_episode(dataset,device)
    else:
        for k in range(len_training_dataset):
            training_dataset.add(dataset[k])
        for k in range(Ntot-len_training_dataset):
            testing_dataset.add(dataset[k+len_training_dataset])

    if shuffle:
        training_dataset = shuffle_dataset(training_dataset)
        testing_dataset = shuffle_dataset(testing_dataset)

    return training_dataset, testing_dataset

def shuffle_dataset(dataset):
    """
    author: alonrot
    """

    # Now, suffle the data:
    dataset_shuffled = SASDataset()
    ind_samples = torch.utils.data.RandomSampler(dataset, replacement=False)
    for ind in ind_samples:
        dataset_shuffled.add(dataset[ind])

    return dataset_shuffled

def store_rng_state(checkpoint, env):
    checkpoint['np_rng_state'] = np.random.get_state()
    checkpoint['torch_rng_state'] = torch.random.get_rng_state()
    checkpoint['env_rng_state'] = env.np_random

    # TODO alonrot:
    if 'torch_cuda_rng_state' in checkpoint:
        checkpoint['torch_cuda_rng_state'] = torch.cuda.set_rng_state()
    else:
        log.warning("checkpoint is missing torch_cuda_rng_state")


def restore_rng_state(checkpoint, env):
    if 'np_rng_state' in checkpoint:
        np.random.set_state(checkpoint['np_rng_state'])
    else:
        log.warning("checkpoint is missing np_rng_state")

    if 'torch_rng_state' in checkpoint:
        torch.set_rng_state(checkpoint['torch_rng_state'])
    else:
        log.warning("checkpoint is missing torch_rng_state")

    if 'torch_cuda_rng_state' in checkpoint:
        torch.cuda.set_rng_state(checkpoint['torch_cuda_rng_state'])
    else:
        log.warning("checkpoint is missing torch_cuda_rng_state")

    if 'env_rng_state' in checkpoint:
        env.np_random = checkpoint['env_rng_state']
    else:
        log.warning("checkpoint is missing env_rng_state")
