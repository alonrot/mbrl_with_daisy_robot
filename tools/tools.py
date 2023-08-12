import sys
import os
import numpy as np
import pdb
import torch
import time
import logging

RAD2DEG = 180. / np.pi
DEG2RAD = 1./RAD2DEG

FIGSIZE = (20,8)

name_motors = ['base1', 'shoulder1', 'elbow1',
                'base2', 'shoulder2', 'elbow2',
                'base3', 'shoulder3', 'elbow3',
                'base4', 'shoulder4', 'elbow4',
                'base5', 'shoulder5', 'elbow5',
                'base6', 'shoulder6', 'elbow6']

name_poses = [  "Position X", "Position Y", "Position Z",
                "Orientation X", "Orientation Y", "Orientation Z", "Orientation W"]

# Pose Position indices:
ind_pose = [18 + 0, 18 + 1, 18 + 2, 18 + 3, 18 + 4, 18 + 5]

class DaisyTools():

    ind_pose_pos = ind_pose[0:3]
    ind_pose_pos_XY = ind_pose[0:2]
    ind_pose_ori = ind_pose[3:6]
    ind_pose = ind_pose

def get_device():
    """
    
    Select cuda / cpu
    """
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print("[WARNING]: USING {0:s} ...".format(device))
    return device

device_global = get_device()

def get_mbrl_base_path():
    """
    
    Get the root mbrl path. When using hydra, os.getcwd() doesn't retirn the root path, 
    but the path where the experiments will be saved. So, we momentarely change the
    path to mbrl, and then change it back.
    os.chdir("../../../")

    author: alonrot

    :return: None
    """

    curr_path = sys.path
    for path_ins in curr_path:
        if path_ins[-4::] == "mbrl":
            break
    return path_ins

    # assert isinstance(using_hydra,bool), "using_hydra = {True,False} (bool)"
    # if using_hydra:
    #     pwd_hydra = os.getcwd() # copy the current cwd to go back to it
    #     os.chdir("../../../") # Go back to the root mbrl path
    #     pwd_mbrl = os.getcwd() # Get the mbrl root path
    #     os.chdir(pwd_hydra) # Go to the hydra path
    # else:
    #     pwd_mbrl = os.getcwd()
    # pdb.set_trace()

    # if pwd_mbrl[-4::] != "mbrl":
    #     pdb.set_trace()
    #     raise ValueError("Something went wrong...")

    # return pwd_mbrl

def get_trajectory_list(traj_init,traj_end,exclude_trajs):

    # Exclude trajectories errorc checking:
    trajectories_list = list(range(traj_init,traj_end))
    if exclude_trajs is not None:

        # Assume that here we are loading a list, which we convert to np array. 
        # Example: 
        #   exclude_trajs: [0,2,34,78]
        #   exclude_trajs: [range(0,19,23,24,range(50,78))]
        exclude_trajs = eval(exclude_trajs)
        assert isinstance(exclude_trajs,list)
        exclude_trajs_new = []
        for exclu in exclude_trajs:
            if isinstance(exclu,range) == True:
                exclude_trajs_new += list(exclu)
            else:
                exclude_trajs_new += exclu
        exclude_trajs = exclude_trajs_new
        assert len(exclude_trajs) <= traj_end - traj_init
        assert len(exclude_trajs) > 0
        exclude_trajs = np.array(exclude_trajs)
        assert np.all(exclude_trajs >= traj_init) and np.all(exclude_trajs <= traj_end-1), "exclude_trajs must be within range(traj_init,traj_end)"
        assert exclude_trajs.dtype == int

        # Exclude trajectories:
        for k in range(len(exclude_trajs)):
            trajectories_list.remove(exclude_trajs[k])
        assert len(trajectories_list) > 0, "You can't exclude all trajectories. traj_init={0:d}, traj_end={1:d}, len(exclude_trajs)={2:d}".format(traj_init,traj_end,len(exclude_trajs))

    return trajectories_list

def load_ep_SAS(path,traj_init,traj_end,exclude_trajs,base_name2save,name_episode=None):

    # Exclude trajectories errorc checking:
    trajectories_list = get_trajectory_list(traj_init,traj_end,exclude_trajs)

    # Put all the episodes together:
    ep_SAS_all = []
    for which_traj in trajectories_list:
        if name_episode is not None:
            path_full = "{0:s}/{1:s}".format(path,name_episode)
            path_full = path_full.replace("xxx",str(which_traj))
            ep_SAS_all += torch.load(path_full,map_location=torch.device(device_global)).episode
        else:
            path_full = path + base_name2save+ "_SAS_episode_"+str(which_traj)+".dat"
            ep_SAS_all += torch.load(path_full,map_location=torch.device(device_global))
    
    Nepi = len(trajectories_list)
    Nsteps_episode = int(len(ep_SAS_all) / Nepi)
    print("Nepi:",Nepi)
    print("Nsteps_episode:",Nsteps_episode)

    return ep_SAS_all, Nepi, Nsteps_episode


def load_data_online_pets(path,analyze_model,device,which_traj):

    assert analyze_model is not None, "analyze_model is not specified in the yaml file"

    # Load data:
    name_folder = analyze_model["name_folder"]
    name_episode = analyze_model["name_episode"]

    # # Exclude trajectories errorc checking:
    # trajectories_list = get_trajectory_list(traj_init,traj_end,exclude_trajs)

    # which_traj = analyze_model["which_traj"]

    # Load entire episodes:
    path_full = "{0:s}/{1:s}/{2:s}".format(path,name_folder,name_episode)
    # for which_traj in trajectories_list:
    #     path_full_ind = path_full.replace("xxx",str(which_traj))
    #     ep_ind = torch.load(path_full_ind,map_location=torch.device(device))

    path_full_ind = path_full.replace("xxx",str(which_traj))
    ep_ind = torch.load(path_full_ind,map_location=torch.device(device))    

    return ep_ind

def load_data4analysis_and_training(cfg):
    """
    Load data that has been collected separately (i.e., without running the actual PETS) on the real robot.
    The purpose is to use this data for training NN and model analysis
    """

    if cfg.data2load.use == "sinewaves_no_floor":
        traj_init = cfg.data2load.sinewaves_no_floor.traj_init
        traj_end = cfg.data2load.sinewaves_no_floor.traj_end
        exclude_trajs = cfg.data2load.sinewaves_no_floor.exclude_trajs
        ind_selector = np.concatenate((np.arange(0,18),np.arange(24,42)))
        path_local = cfg.data2load.sinewaves_no_floor.path
        model_name = cfg.data2load.sinewaves_no_floor.model_name
        remove_unwanted_states = cfg.data2load.sinewaves_no_floor.remove_unwanted_states
        folder2model = cfg.data2load.sinewaves_no_floor.folder2model
        base_name2save = cfg.data2load.sinewaves_no_floor.base_name2save
        normalize_data = cfg.data2load.sinewaves_no_floor.normalize_data
        analyze_model = None
    elif cfg.data2load.use == "random_no_vel":
        traj_init = cfg.data2load.random_no_vel.traj_init
        traj_end = cfg.data2load.random_no_vel.traj_end
        exclude_trajs = cfg.data2load.random_no_vel.exclude_trajs
        ind_selector = np.arange(0,18)
        path_local = cfg.data2load.random_no_vel.path
        model_name = cfg.data2load.random_no_vel.model_name
        remove_unwanted_states = cfg.data2load.random_no_vel.remove_unwanted_states
        folder2model = cfg.data2load.random_no_vel.folder2model
        base_name2save = cfg.data2load.random_no_vel.base_name2save
        normalize_data = cfg.data2load.random_no_vel.normalize_data
        analyze_model = None
    elif cfg.data2load.use == "cpg":
        traj_init = cfg.data2load.cpg.traj_init
        traj_end = cfg.data2load.cpg.traj_end
        exclude_trajs = cfg.data2load.cpg.exclude_trajs
        # ind_selector = np.arange(0,42)
        ind_selector = np.concatenate((np.arange(0,18),np.arange(18,20),np.arange(24,42)))
        path_local = cfg.data2load.cpg.path
        model_name = cfg.data2load.cpg.model_name
        remove_unwanted_states = cfg.data2load.cpg.remove_unwanted_states
        folder2model = cfg.data2load.cpg.folder2model
        base_name2save = cfg.data2load.cpg.base_name2save
        normalize_data = cfg.data2load.cpg.normalize_data
        analyze_model = dict(   name_folder=cfg.data2load.cpg.analyze_model.name_folder,
                                name_episode=cfg.data2load.cpg.analyze_model.name_episode)
    else:
        raise ValueError("'Incorrect cfg.data2load.use' cfg.data2load.use={0:s}".format(cfg.data2load.use))

    # Get mbrl root path, as the data paths are relative to this one:
    path = "{0:s}/{1:s}".format(get_mbrl_base_path(),path_local)
    log.info("Loading data from {0:s}".format(path))
    
    # Load data:
    ep_SAS_all, Nepi, Nsteps_episode = load_ep_SAS(path,traj_init,traj_end,exclude_trajs,base_name2save)

    # Append walking data, if desired by the user:
    assert cfg.data2load.use == "cpg", "This needs to be preoperly done"
    if cfg.data2load.cpg.analyze_model.append_walking_data_for_offline_training.use:
        path_walking = cfg.data2load.cpg.analyze_model.append_walking_data_for_offline_training.name_folder
        path = "{0:s}/{1:s}".format(path,path_walking)
        traj_init = cfg.data2load.cpg.analyze_model.append_walking_data_for_offline_training.traj_init
        traj_end = cfg.data2load.cpg.analyze_model.append_walking_data_for_offline_training.traj_end
        exclude_trajs = cfg.data2load.cpg.analyze_model.append_walking_data_for_offline_training.exclude_trajs
        # base_name2save = cfg.data2load.cpg.analyze_model.append_walking_data_for_offline_training.base_name2save
        name_episode = cfg.data2load.cpg.analyze_model.append_walking_data_for_offline_training.name_episode
        ep_SAS_all_walking, Nepi, Nsteps_episode = load_ep_SAS(path,traj_init,traj_end,exclude_trajs,"",name_episode)
        ep_SAS_all += ep_SAS_all_walking

    # Create a dictionary:
    out = dict( traj_init=traj_init,traj_end=traj_end,exclude_trajs=exclude_trajs,ind_selector=ind_selector,path_local=path_local,
                model_name=model_name,remove_unwanted_states=remove_unwanted_states,folder2model=folder2model,
                path=path,ep_SAS_all=ep_SAS_all,Nepi=Nepi,Nsteps_episode=Nsteps_episode,
                base_name2save=base_name2save,normalize_data=normalize_data,analyze_model=analyze_model)

    return out

def load_dynamics_model(path,folder2model,model_name,device):
    """

    Wrapper to load the dynamics model.
    """

    # Load the trained model:
    (dynamics_model,training_log) = torch.load(path+folder2model+model_name,map_location=torch.device(device))
    # assert isinstance(dynamics_model, NNBasedDynamicsModel) # TODO alonrot: Add this!
    time_training_tot = training_log["time_tot"]
    log.info("Total training time: {0:5.2f} [sec]".format(time_training_tot))

    if "epochs" in training_log.keys():
        Nr_epochs = training_log["epochs"]
        log.info("Nr. epochs:          {0:d}".format(Nr_epochs))
        log.info("time/epoch:          {0:2.2f} [sec]".format(time_training_tot/Nr_epochs))

    # Change the internal device of the entire class to the specified one:
    dynamics_model.change_internal_device_to(device)
    # dynamics_model.set_jit(jit) # Taken from pets.main.init() | Maybe it's unnecessary as torch.load() already calls set_jit()
    # The line above is unnecessary because self.jit_model() is already called when loading the model with torch

    # # Backwards compatibility *(unsupported):
    # dynamics_model = torch.load("{0:s}/{1:s}".format(path+folder2model,model_name),map_location=torch.device(device))
    # use_class_model = False



    return dynamics_model

def normalize_episodes(ep_SAS_all,ep_SAS_all_mean=None,ep_SAS_all_std=None,check_std_is_nonzero=True):
    r"""
    Modify the given data set to have zero mean and unit variance across all states and actions.
    author: alonrot
    
    :param ep_SAS_all: input dataset; typically a collected of SAS (state-action-state) data points
    from experiments on the robot.
    :type ep_SAS_all: mbrl.SASDataset

    :return ep_SAS_all: Modified given data set to have zero mean and unit variance
    :type ep_SAS_all: mbrl.SASDataset

    .. note::

        This function does not return anything, as it modified the memory of ep_SAS_all

    """

    # Initialize:
    Ntot = len(ep_SAS_all)
    dim_s0 = len(ep_SAS_all[0].s0)
    dim_s1 = len(ep_SAS_all[0].s1)
    dim_a = len(ep_SAS_all[0].a)
    device = ep_SAS_all[0].s0.device
    s0_all = torch.zeros((Ntot,dim_s0),device=device)
    s1_all = torch.zeros((Ntot,dim_s1),device=device)
    a_all = torch.zeros((Ntot,dim_a),device=device)

    for k in range(Ntot):
        s0_all[k,:] = ep_SAS_all[k].s0
        s1_all[k,:] = ep_SAS_all[k].s1
        a_all[k,:] = ep_SAS_all[k].a

    if ep_SAS_all_mean is None:
        s0_mean = torch.mean(s0_all,dim=0)
        s1_mean = torch.mean(s1_all,dim=0)
        a_mean = torch.mean(a_all,dim=0)
    else:
        assert isinstance(ep_SAS_all_mean,list)
        s0_mean = ep_SAS_all_mean[0]
        a_mean = ep_SAS_all_mean[1]
        s1_mean = ep_SAS_all_mean[2]

    if ep_SAS_all_std is None:
        s0_std = torch.std(s0_all,dim=0)
        s1_std = torch.std(s1_all,dim=0)
        a_std = torch.std(a_all,dim=0)
    else:
        assert isinstance(ep_SAS_all_std,list)
        s0_std = ep_SAS_all_std[0]
        a_std = ep_SAS_all_std[1]
        s1_std = ep_SAS_all_std[2]

    if check_std_is_nonzero:
        try:
            assert not torch.any(s0_std == 0.0), "No std can be zero"
            assert not torch.any(s1_std == 0.0), "No std can be zero"
            assert not torch.any(a_std == 0.0), "No std can be zero"
        except Exception as e:
            print(e,type(e))
            pdb.set_trace()

    for k in range(Ntot):
        ep_SAS_all[k].s0 = (ep_SAS_all[k].s0 - s0_mean) / s0_std
        ep_SAS_all[k].s1 = (ep_SAS_all[k].s1 - s1_mean) / s1_std
        ep_SAS_all[k].a = (ep_SAS_all[k].a - a_mean) / a_std

    return [s0_mean,a_mean,s1_mean], [s0_std,a_std,s1_std]

def unnormalize_episodes(ep_SAS_all,ep_SAS_all_mean,ep_SAS_all_std):
    r"""
    Modify the given data set to have zero mean and unit variance across all states and actions.
    author: alonrot
    
    :param ep_SAS_all: input dataset; typically a collected of SAS (state-action-state) data points
    from experiments on the robot.
    :type ep_SAS_all: mbrl.SASDataset

    :return ep_SAS_all: Modified given data set to have zero mean and unit variance
    :type ep_SAS_all: mbrl.SASDataset

    .. note::

        This function does not return anything, as it modified the memory of ep_SAS_all

    """

    # Initialize:
    Ntot = len(ep_SAS_all)
    dim_s0 = len(ep_SAS_all[0].s0)
    dim_s1 = len(ep_SAS_all[0].s1)
    dim_a = len(ep_SAS_all[0].a)

    assert isinstance(ep_SAS_all_mean,list)
    s0_mean = ep_SAS_all_mean[0]
    a_mean = ep_SAS_all_mean[1]
    s1_mean = ep_SAS_all_mean[2]

    assert isinstance(ep_SAS_all_std,list)
    s0_std = ep_SAS_all_std[0]
    a_std = ep_SAS_all_std[1]
    s1_std = ep_SAS_all_std[2]

    assert not torch.any(s0_std == 0.0), "No std can be zero"
    assert not torch.any(s1_std == 0.0), "No std can be zero"
    assert not torch.any(a_std == 0.0), "No std can be zero"

    for k in range(Ntot):
        ep_SAS_all[k].s0 = ep_SAS_all[k].s0*s0_std + s0_mean
        ep_SAS_all[k].s1 = ep_SAS_all[k].s1*s1_std + s1_mean
        ep_SAS_all[k].a = ep_SAS_all[k].a*a_std + a_mean

    return

def create_file_name2save(cfg):

    # Changing parameters:
    batch_size      = cfg.training.batch_size
    full_epochs     = cfg.training.full_epochs
    shuffle_data    = cfg.training.shuffle_data
    lr              = cfg.training.optimizer.params.lr

    # Other parameters:
    assert cfg.data2load.use == "cpg"
    normalize_data  = cfg.data2load.cpg.normalize_data
    split           = cfg.training.testing.split
    seed            = cfg.random_seed

    # Fixed parameters:
    name_base       = cfg.training.name_base
    device          = cfg.device
    Nstates         = cfg.env.state_size

    # file_name = "{6:s}_batch{0:d}_epochs{1:d}_shuffle{2:s}_lr{3:.1E}_device_{4:s}_Nstates{5:d}".format(batch_size,full_epochs,str(shuffle_data),lr,device,Nstates,name_base)
    file_name = "{6:s}_batch{0:d}_epochs{1:d}_shuffle{2:s}_lr{3:.1E}_device_{4:s}_Nstates{5:d}_norm{7:s}_split{8:1.1f}_seed{9:d}".format(batch_size,full_epochs,str(shuffle_data),lr,device,Nstates,name_base,str(normalize_data),split,seed)

    return file_name

def fix_pose_for_model_training(ep_SAS_all, Nepi, Nsteps_episode):
    """
    
    Modify the real data to make sure that the starting position is all the same for all the episodes
    In the future, this will be done on-line, while doing experiments, and this fucntion will be no longer required.

    This function does not return, as it just modifies the list ep_SAS_all passed as input
    """

    # Pose Position indices:
    ind_pose_X = ind_pose[0]
    ind_pose_Y = ind_pose[1]
    ind_pose_Z = ind_pose[2]

    for ind_init_epi in range(0,Nepi*Nsteps_episode,Nsteps_episode):
        s0_X_init = ep_SAS_all[ind_init_epi].s0[ind_pose_X].item()
        s0_Y_init = ep_SAS_all[ind_init_epi].s0[ind_pose_Y].item()
        
        s1_X_init = ep_SAS_all[ind_init_epi].s1[ind_pose_X].item()
        s1_Y_init = ep_SAS_all[ind_init_epi].s1[ind_pose_Y].item()

        # pdb.set_trace()
        for k in range(Nsteps_episode):
            ep_SAS_all[ind_init_epi+k].s0[ind_pose_X] -= s0_X_init
            ep_SAS_all[ind_init_epi+k].s0[ind_pose_Y] -= s0_Y_init
            
            ep_SAS_all[ind_init_epi+k].s1[ind_pose_X] -= s1_X_init
            ep_SAS_all[ind_init_epi+k].s1[ind_pose_Y] -= s1_Y_init

    return

def remove_unwanted_states_func(ep_SAS_all,ind_selector):
    """

    Remove states, by leaving only those specified in ind_selector
    """
    print("Removing unwanted states...")
    print("ind_selector:",ind_selector)
    Nstates = len(ep_SAS_all[0].s0)
    print("Nstates = {0:d}".format(Nstates))
    print("Removing...")
    for SAS_el in ep_SAS_all:
        SAS_el.s0 = SAS_el.s0[ind_selector]
        SAS_el.s1 = SAS_el.s1[ind_selector]
    Nstates = len(ep_SAS_all[0].s0)
    assert Nstates == len(ind_selector)
    print("Nstates = {0:d}".format(Nstates))

    # pdb.set_trace()
    if ep_SAS_all[0].a.dtype == torch.float64:
        for SAS_el in ep_SAS_all:
            SAS_el.a = SAS_el.a.type(torch.float32)
    # pdb.set_trace()

    return Nstates

log = logging.getLogger(__name__)
def create_new_folder(path_and_folder,ask_user=True,append_timeofday=False):

    # Append the time of the day, if requested by the user:
    if append_timeofday:
        timeofday_append_str = time.strftime("/%Y_%m_%d/%H_%M_%S/")
        path_and_folder += timeofday_append_str

    if not os.path.isdir(path_and_folder):
        log.info("Creating a new location to store data inside the folder where the model was loaded from:")
        log.info(path_and_folder)
        log.info("does not exist. Do you want to create it? [0,1]: ")
        aux = 999
        while aux != "1" and aux != "0":
            if ask_user: 
                log.info("Enter 1 or 0: ")
                aux = input()
            else:
                aux = "1"
        if aux == "1":
            try:
                os.makedirs(path_and_folder,exist_ok=False)
            except:
                pdb.set_trace()
        folder_created = aux == "1"
    else:
        folder_created = True

    return path_and_folder, folder_created

def acquire_rewards(ep):

    Nrewards = len(ep.rewards)
    rewards_vec = np.zeros(Nrewards)

    for k in range(Nrewards):
        rewards_vec[k] = ep.rewards[k][0] # Take [0] because it's a vector with one element

    return rewards_vec

def acquire_state(ep,which_ind):

    Nrewards = len(ep.episode)
    state_vec = np.zeros((Nrewards,len(which_ind)))

    for k in range(Nrewards):
        state_vec[k,:] = ep.episode[k].s0[which_ind].cpu().numpy()

    # import pdb; pdb.set_trace()

    return state_vec


# def ask_user_input(msg_input):

#     assert isinstance(msg_input,str)

#     ipt = 999
#     while ipt != 0 and ipt != 1:
#         ipt = input(msg_input)
#         ipt = int(ipt)

#     if ipt == 1:
#         s.write_async_data_output_frequency(freq_new)
#         print("Output frequency successfully changed!")
#         print("New output frequency: {0:d} Hz".format(freq_new))
#     else:
#         print("Output frequency not changed!")



class GymBox():
    """
    Other parts of mbrl and pets, like the cem optimizer, are coded using gym functionalities, like:
        from gym import spaces
    This should change, as the real robot code is not tight to gym
        self.action_space = spaces.Box(-1.0, 1.0, shape=[int(len(self.robot.ordered_joints))], dtype=np.float32)
    
    This function is a temporary fix to this issue that avoids importing gym, until the lowe level code
    issues are solved. This function mimics the functions from gym.spaces that are needed
    """

    def __init__(self, lim_low, lim_high, length):
        
        self.low    = lim_low*np.ones(length, dtype=np.float32)
        self.high   = lim_high*np.ones(length, dtype=np.float32)


        # # Limit the joint angles of the bases:
        # if length == 18:
        #     ind_base = np.arange(0,18,3)
        #     self.low[ind_base]  = -10.*DEG2RAD
        #     self.high[ind_base] = +10.*DEG2RAD

        self.shape = (length,)

        self.ind_box_dict = None

    def __len__(self):
        return self.shape[0]

    @classmethod
    def create_from_vec(cls, low_vec, high_vec, ind_box_dict=None):

        # Check for errors:
        assert isinstance(low_vec,np.ndarray)
        assert low_vec.ndim == 1
        assert isinstance(high_vec,np.ndarray)
        assert high_vec.ndim == 1
        assert len(low_vec) == len(high_vec)

        # Create the class object with dummy parameters:
        cls_obj = cls(0.0,1.0,len(low_vec))

        # Modify its members:
        cls_obj.low[:] = low_vec
        cls_obj.high[:] = high_vec

        if ind_box_dict is not None:
            cls_obj.ind_box_dict = ind_box_dict

        return cls_obj
