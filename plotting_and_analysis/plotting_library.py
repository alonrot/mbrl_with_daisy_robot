import matplotlib.pyplot as plt
import matplotlib.cm as colormap
import numpy as np
import numpy.linalg as la
import torch
from datetime import datetime
import pdb
import hydra
import sys
import os
import _pickle
import matplotlib
from tools.tools import RAD2DEG, FIGSIZE, name_motors, name_poses, \
                        ind_pose, get_device, load_data_online_pets, \
                        normalize_episodes, get_trajectory_list, get_mbrl_base_path, load_ep_SAS,\
                        load_dynamics_model
device_global = get_device()
import logging

# For trajectory propagation:
from mbrl.trajectories.simple_prop import SimpleProp
from mbrl.trajectories.tsprop import TSProp
# from mbrl.rewards.reward_base import Reward4NNanalysis
import mbrl.rewards

log = logging.getLogger(__name__)

# matplotlib.rc('text', usetex=True)
# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans Serif'],'size': 20})
# matplotlib.rc('xtick', labelsize=18)
# matplotlib.rc('ytick', labelsize=18) 

def plot_pose_XY(cfg_pose_XY,ep_SAS_all,Nepi,Nsteps_episode,path2save=None):
    """

    Plot position in the X-Y plane, and the height Z
    """

    colors = colormap.get_cmap("gnuplot2") # Short in colors for 10 episodes
    # colors = colormap.get_cmap("nipy_spectral")
    # colors = colormap.get_cmap("gist_rainbow") # Too colorful
    colors_mat = np.zeros((Nepi,3))
    for k in range(Nepi):
        colors_mat[k,:] = colors(k/Nepi)[0:3]

    # Generate action sequence:
    action_sequence, state_curr_vec, state_next_vec, time_vec = generate_action_sequence(cfg_pose_XY,ep_SAS_all)

    # Pose Position indices:
    ind_pose_X = ind_pose[0]
    ind_pose_Y = ind_pose[1]
    ind_pose_Z = ind_pose[2]

    # Get pose:
    pose_X_vec = state_curr_vec[:,ind_pose_X]
    pose_Y_vec = state_curr_vec[:,ind_pose_Y]
    pose_Z_vec = state_curr_vec[:,ind_pose_Z]

    hdl_fig, hdl_plot = plt.subplots(2,1,figsize=(16,16))
    hdl_fig.suptitle("Robot position - Central Pattern Generator (CPG) - 20 episodes")
    n_epi = 0
    for k in range(0,Nepi*Nsteps_episode,Nsteps_episode):
        pose_X_vec_epi = pose_X_vec[k:k+Nsteps_episode]
        pose_Y_vec_epi = pose_Y_vec[k:k+Nsteps_episode]
        hdl_plot[0].plot(pose_X_vec_epi,pose_Y_vec_epi,label="poseXY - Episode {0:d}".format(n_epi+1),linestyle="--",color=colors_mat[n_epi,:],marker=".")
        n_epi += 1
    hdl_plot[0].set_xlabel("X [m]")
    hdl_plot[0].set_ylabel("Y [m]")
    # hdl_plot[0].legend()
    hdl_plot[0].set_title("Position in X-Y plane")

    hdl_plot[1].plot(time_vec,pose_Z_vec,label="poseZ",linestyle="--",color="blue",marker=".")
    # hdl_plot[1].legend()
    hdl_plot[1].set_xlabel("time [sec]")
    hdl_plot[1].set_ylabel("Z [m]")
    hdl_plot[1].set_title("Height Z")

    plt.show(block=cfg_pose_XY.block)

    if cfg_pose_XY.path2save is not None:
        # pdb.set_trace()
        # plt.tight_layout()
        plt.savefig(cfg_pose_XY.path2save,dpi=100)

def plot_orientation(cfg_pose_XY,ep_SAS_all,Nepi,Nsteps_episode):
    pass

def plot_SAS(cfg_SAS,ep_SAS_all,ind_state_local,ind_action):

    # Generate action sequence:
    action_sequence, state_curr_vec, state_next_vec, time_vec = generate_action_sequence(cfg_SAS,ep_SAS_all)

    # Select desired state index:
    ind_state, ind_action, scale_signal4plotting, ylabel_state = parse_user_specified_indices(cfg_signal=cfg_SAS,Nstates=len(ep_SAS_all[0].s0))

    s0_vec = state_curr_vec[:,ind_state]*scale_signal4plotting
    s1_vec = state_next_vec[:,ind_state]*scale_signal4plotting
    ac_vec = action_sequence[:,ind_action]*RAD2DEG

    hdl_fig, hdl_plot = plt.subplots(2,1,figsize=FIGSIZE,sharex=True)
    hdl_fig.suptitle("Raw data analysis")
    hdl_plot[0].plot(s0_vec,label="state[k]",linestyle="--",color="brown",marker=".")
    hdl_plot[0].plot(s1_vec,label="state[k+1]",linestyle="--",color="darkgreen",marker=".")
    # hdl_plot[0].plot(ac_vec,label="action[k]",linestyle="-",color="brown",marker=".")
    hdl_plot[0].set_ylabel(ylabel_state)
    hdl_plot[0].legend()
    hdl_plot[0].set_title("State {0:s} | local index {1:d}".format(cfg_SAS.name_state2visualize,cfg_SAS.ind_state_local))
    hdl_plot[0].set_facecolor("whitesmoke")

    hdl_plot[1].plot(ac_vec,linestyle="--",color="purple",marker=".")
    hdl_plot[1].set_title("Applied action in {0:s}".format(name_motors[ind_action]))
    hdl_plot[1].set_ylabel("Angle [deg]")
    hdl_plot[1].set_facecolor("whitesmoke")
    
    hdl_plot[1].set_xlabel("Niters")

    plt.show(block=cfg_SAS.block)

def action_sequence_rollout_to_predict_state_sequence(cfg_predict_rollout,ep_SAS_all,my_model):

    # Generate action sequence:
    action_sequence, state_curr_vec, state_next_vec, time_vec = generate_action_sequence(cfg_predict_rollout,ep_SAS_all)

    # Roll-out trained model from step_init until step_end:
    Nsteps = action_sequence.shape[0]
    Nstates = state_curr_vec.shape[1]
    delta_state_np_predicted = np.zeros((Nsteps+1,Nstates,2),dtype=np.float32)
    mean_next_state_cumsum_predicted = np.zeros((Nsteps+1,Nstates),dtype=np.float32)
    var_next_state_cumsum_predicted = np.zeros((Nsteps+1,Nstates),dtype=np.float32)
    mean_next_state_cumsum_predicted[0,:] = state_curr_vec[0,:]
    # Nsteps = 10
    for k in range(Nsteps-1):
        
        # Prepare inputs:
        state_np = mean_next_state_cumsum_predicted[k,:]
        action_np = action_sequence[k,:]
            
        state_input = torch.from_numpy(state_np[None,:]).to(device_global)
        action_input = torch.from_numpy(action_np[None,:]).to(device_global)
        
        # Prediction:
        try:
            y_output_particles = my_model.predict(state_input,action_input) # The NN predicts deltas, but .predict() resturns the absolute state. The input is the absolute state.
        except:
            pdb.set_trace()

        # Average over number of particles:
        y_output = y_output_particles.mean(dim=2)
        # y_output = y_output_particles[:,:,0,:]

        # Convert to numpy:
        y_output_np = y_output[0,:,:].cpu().detach().numpy()

        # Propagate:
        mean_next_state_cumsum_predicted[k+1,:] = y_output_np[:,0]
        # pdb.set_trace()

        # Calculate deltas:
        delta_state_np_predicted[k+1,:,0] = mean_next_state_cumsum_predicted[k+1,:] - mean_next_state_cumsum_predicted[k,:] # Isn't this delta_state_np_predicted[k+1,:,0]
        delta_state_np_predicted[k+1,:,1] = y_output_np[:,1]

        var_next_state_cumsum_predicted[k+1,:] = y_output_np[:,1] + var_next_state_cumsum_predicted[k,:]

    # pdb.set_trace()

    # Correct shift by deleting the first element (it's correct, double-checked twice)
    mean_next_state_cumsum_predicted = np.delete(mean_next_state_cumsum_predicted,0,axis=0)
    delta_state_np_predicted = np.delete(delta_state_np_predicted,0,axis=0)
    var_next_state_cumsum_predicted = np.delete(var_next_state_cumsum_predicted,0,axis=0)

    # mean_cumsum_manual = np.cumsum(delta_state_np_predicted[:,cfg_predict_rollout.which_joint,0])

    # Select desired state index:
    ind_state, ind_action, scale_signal4plotting, ylabel_state = parse_user_specified_indices(cfg_signal=cfg_predict_rollout,Nstates=len(ep_SAS_all[0].s0))

    state_curr_vec_sel_joint            = state_curr_vec[:,ind_state]
    state_next_vec_sel_joint            = state_next_vec[:,ind_state]
    delta_state_np_predicted_sel_joint  = delta_state_np_predicted[:,ind_state,0]
    action_vec_sel_joint                = action_sequence[:,ind_action]
    mean_next_state_cumsum_predicted_sel_joint = mean_next_state_cumsum_predicted[:,ind_state]
    var_next_state_cumsum_predicted_sel_joint = var_next_state_cumsum_predicted[:,ind_state]

    std_plus_mean_next_state_cumsum_predicted_sel_joint = mean_next_state_cumsum_predicted_sel_joint + np.sqrt(var_next_state_cumsum_predicted_sel_joint)
    std_minus_mean_next_state_cumsum_predicted_sel_joint = mean_next_state_cumsum_predicted_sel_joint - np.sqrt(var_next_state_cumsum_predicted_sel_joint)


    std_plus_mean_p = delta_state_np_predicted[:,ind_state,0] + np.sqrt(delta_state_np_predicted[:,ind_state,1])
    std_plus_mean_m = delta_state_np_predicted[:,ind_state,0] - np.sqrt(delta_state_np_predicted[:,ind_state,1])

    # Look at:
    # 1) Structure of the NN
    # 2) Try to save the NN, and load it using pytorch

    # Compute one-step prediction error:
    err_next_step = (state_next_vec_sel_joint - mean_next_state_cumsum_predicted_sel_joint)**2

    # # Original plots:
    # add_plot_action = cfg_predict_rollout.name_state2visualize in ["joint_angular_pos","joint_angular_vel"]
    # n_splots = 2 + int(add_plot_action)
    # hdl_fig, hdl_plot = plt.subplots(n_splots,1,figsize=FIGSIZE,sharex=True)
    # hdl_fig.suptitle("Trained model roll-out; departing from a specific time step; using the true ations")
    
    # if cfg_predict_rollout.use == "original":
    #     hdl_plot[0].plot(time_vec,state_curr_vec_sel_joint*scale_signal4plotting,label="state_curr_data",linestyle="--",color="cornflowerblue",marker=".")
    #     hdl_plot[0].plot(time_vec,state_next_vec_sel_joint*scale_signal4plotting,label="state_next_data",linestyle="--",color="brown",marker=".")
    # else:
    #     hdl_plot[0].plot(time_vec,action_vec_sel_joint*RAD2DEG,label="action",linestyle="-",color="brown",marker=".")
    
    # hdl_plot[0].plot(time_vec,mean_next_state_cumsum_predicted_sel_joint*scale_signal4plotting,label="state_predicted_next_absolute",linestyle="-",color="lightcoral",marker="None",linewidth=1.5)
    # hdl_plot[0].plot(time_vec,delta_state_np_predicted_sel_joint*scale_signal4plotting,label="state_predicted_next_delta_mean",linestyle="-",color="green",marker="None",linewidth=1.5)
    # hdl_plot[0].plot(time_vec,std_plus_mean_p*scale_signal4plotting,label="state_predicted_next_delta_std_p",linestyle=":",color="green",marker="None",linewidth=1.5)
    # hdl_plot[0].plot(time_vec,std_plus_mean_m*scale_signal4plotting,label="state_predicted_next_delta_std_m",linestyle=":",color="green",marker="None",linewidth=1.5)
    # # hdl_plot[0].set_ylim([-75.,+75.])
    # # hdl_plot[0].set_xlim([0.,30.])
    # hdl_plot[0].legend(loc="lower right")
    # hdl_plot[0].set_title("State {0:s} | local index {1:d}".format(cfg_predict_rollout.name_state2visualize,cfg_predict_rollout.ind_state_local))
    # hdl_plot[0].set_ylabel(ylabel_state)

    # hdl_plot[1].plot(time_vec,err_next_step*scale_signal4plotting,linestyle="-",color="violet",marker=".")
    # hdl_plot[1].set_xlabel("time [sec]")
    # hdl_plot[1].set_ylabel("Squared error")

    # if add_plot_action:
    #     hdl_plot[2].plot(time_vec,action_vec_sel_joint*RAD2DEG,linestyle="-",color="brown",marker=".")
    #     hdl_plot[2].set_title("Applied action; Joint: {0:s}".format(name_motors[ind_action]))
    #     hdl_plot[2].set_xlabel("time [sec]")
    #     hdl_plot[2].set_ylabel("State [deg]")


    label_state_next_pred = r"$x_{t+1}$ (prediction)"
    label_state_next_groundtruth = r"$\hat{x}_{t+1}$ (groundtruth)"

    title_delta_pred = r"Predicted distribution over delta state $p(\Delta x_{t+1} | \hat{x}_t, \hat{a}_t)$"
    title_absolute_state_pred = "State propagation"

    # if cfg_one_step.name_state2visualize == "joint_angular_pos":
    #     name_state2visualize = "Joint angular position"
    # else:
    #     name_state2visualize = cfg_one_step.name_state2visualize

    # if cfg_one_step.ind_state_local == 8:
    #     ind_state_local = "Elbow 3"
    # else:
    #     ind_state_local = cfg_one_step.ind_state_local

# hdl_fig.suptitle("Quality of next-state predictions: $x_{t+1} \sim p(\Delta x_{t+1} | \hat{x}_t, \hat{a}_t) + \hat{x}_t$")
# hdl_plot.set_title("Next state (sorted) | {0:s} | {1:s}".format(name_state2visualize,ind_state_local))

    if ind_state == 8:
        which_joint = "Elbow 3"
        ylim_rollout = [-120.,-40.]
        loc_legend = "upper right"
    elif ind_state == 18:
        which_joint = "Position X"
        ylim_rollout = [-0.1,0.9]
        loc_legend = "upper left"
    else:
        which_joint = str(ind_state)
        ylim_rollout = None
        loc_legend = None


    # Plotting for presentation:
    add_plot_action = cfg_predict_rollout.name_state2visualize in ["joint_angular_pos","joint_angular_vel"]
    hdl_fig, hdl_plot = plt.subplots(2,1,figsize=(13,8),sharex=True)
    # hdl_fig.suptitle("Trained model roll-out; departing from a specific time step; using the true ations")
    hdl_fig.suptitle("Quality of long-term predictions | {0:s}".format(which_joint))
    
    if cfg_predict_rollout.use == "original":
        # hdl_plot[0].plot(time_vec,state_curr_vec_sel_joint*scale_signal4plotting,label="state_curr_data",linestyle="--",color="cornflowerblue",marker=".")
        hdl_plot[0].plot(time_vec,state_next_vec_sel_joint*scale_signal4plotting,label=label_state_next_groundtruth,linestyle="--",color="cornflowerblue",marker=".")
    else:
        hdl_plot[0].plot(time_vec,action_vec_sel_joint*RAD2DEG,label="action",linestyle="-",color="brown",marker=".")
    
    xlim = [time_vec[0],time_vec[160]]
    hdl_plot[0].fill_between(time_vec, std_minus_mean_next_state_cumsum_predicted_sel_joint*scale_signal4plotting, std_plus_mean_next_state_cumsum_predicted_sel_joint*scale_signal4plotting,color='lightcoral', alpha=0.2,label="std")
    hdl_plot[0].plot(time_vec,mean_next_state_cumsum_predicted_sel_joint*scale_signal4plotting,label=label_state_next_pred,linestyle="-",color="lightcoral",marker="None",linewidth=2.0)
    # hdl_plot[0].plot(time_vec,std_plus_mean_next_state_cumsum_predicted_sel_joint*scale_signal4plotting,label="state_predicted_next_absolute",linestyle="-",color="lightcoral",marker="None",linewidth=1.0)
    # hdl_plot[0].plot(time_vec,std_minus_mean_next_state_cumsum_predicted_sel_joint*scale_signal4plotting,label="state_predicted_next_absolute",linestyle="-",color="lightcoral",marker="None",linewidth=1.0)
    # hdl_plot[0].plot(time_vec,delta_state_np_predicted_sel_joint*scale_signal4plotting,label="state_predicted_next_delta_mean",linestyle="-",color="green",marker="None",linewidth=1.5)
    hdl_plot[0].set_ylim(ylim_rollout)
    hdl_plot[0].set_xlim(xlim)
    hdl_plot[0].legend(loc=loc_legend)
    hdl_plot[0].set_title(title_absolute_state_pred)
    hdl_plot[0].set_ylabel(ylabel_state)

    # pdb.set_trace()
    hdl_plot[1].fill_between(time_vec, std_plus_mean_m*scale_signal4plotting, std_plus_mean_p*scale_signal4plotting,color='green', alpha=0.2,label="std")
    hdl_plot[1].plot(time_vec,delta_state_np_predicted_sel_joint*scale_signal4plotting,label="mean",linestyle="-",color="green",marker="None",linewidth=1.5)
    # hdl_plot[1].plot(time_vec,std_plus_mean_p*scale_signal4plotting,label="state_predicted_next_delta_std_p",linestyle=":",color="green",marker="None",linewidth=1.5)
    # hdl_plot[1].plot(time_vec,std_plus_mean_m*scale_signal4plotting,label="state_predicted_next_delta_std_m",linestyle=":",color="green",marker="None",linewidth=1.5)
    # hdl_plot[1].set_ylim([-75.,+75.])
    hdl_plot[1].set_xlim(xlim)
    hdl_plot[1].legend(loc="lower right")
    hdl_plot[1].set_title(title_delta_pred)
    hdl_plot[1].set_ylabel(ylabel_state)
    hdl_plot[1].set_xlabel("time [sec]")

    plt.show(block=cfg_predict_rollout.block)

def one_step_predictions_sorted(cfg_one_step,ep_SAS_all,my_model):

    action_sequence, state_curr_vec, state_next_vec, time_vec = generate_action_sequence(cfg_one_step,ep_SAS_all)

    # Roll-out trained model from step_init until step_end:
    Nsteps = action_sequence.shape[0]
    Nstates = state_curr_vec.shape[1]

    # Roll-out trained model from step_init until step_end:
    delta_state_np_predicted = np.zeros((Nsteps,Nstates,2),dtype=np.float32)
    absolute_state_np_predicted = np.zeros((Nsteps,Nstates),dtype=np.float32)
    for k in range(Nsteps-1):

        state_input_np = state_curr_vec[k,:]
        action_input_np = action_sequence[k,:]
        state_input = torch.from_numpy(state_input_np[None,:]).to(device_global)
        action_input = torch.from_numpy(action_input_np[None,:]).to(device_global)
        # pdb.set_trace()
        
        # Prediction:
        y_output_particles = my_model.predict(state_input,action_input)
        # try:
        #     y_output_particles = my_model.predict(state_input,action_input)
        # except:
        #     pdb.set_trace()

        # Average over number of particles:
        y_output = y_output_particles.mean(dim=2)
        # y_output = y_output_particles[:,:,0,:]
        # pdb.set_trace()

        # Convert to numpy:
        absolute_state_np_predicted[k,:] = y_output[0,:,0].cpu().detach().numpy()

        if k > 0:
            delta_state_np_predicted[k,:,0] = absolute_state_np_predicted[k,:] - absolute_state_np_predicted[k-1,:]
            delta_state_np_predicted[k,:,1] = y_output[0,:,1].cpu().detach().numpy()

    # Select desired state index:
    ind_state, ind_action, scale_signal4plotting, ylabel_state = parse_user_specified_indices(cfg_signal=cfg_one_step,Nstates=len(ep_SAS_all[0].s0))

    # Select the corresponding state and action:
    absolute_state_np_predicted_sel_joint = absolute_state_np_predicted[:,ind_state]
    state_groundtruth_sel_joint = state_next_vec[:,ind_state]
    # state_groundtruth_sel_joint = state_curr_vec[:,ind_state]
    action_vec_sel_joint = action_sequence[:,ind_action]


    # Shorten:
    assert len(state_groundtruth_sel_joint) > 1
    assert len(absolute_state_np_predicted_sel_joint) > 1

    state_groundtruth_sel_joint = state_groundtruth_sel_joint[0:-1]
    absolute_state_np_predicted_sel_joint = absolute_state_np_predicted_sel_joint[0:-1]

    # Sort according to groundtruth:
    ind_sort = np.argsort(state_groundtruth_sel_joint)
    # ind_sort = np.arange(0,len(state_groundtruth_sel_joint))
    state_groundtruth_sel_joint_sorted = state_groundtruth_sel_joint[ind_sort]
    absolute_state_np_predicted_sel_joint_sorted = absolute_state_np_predicted_sel_joint[ind_sort]
    action_vec_sel_joint_sorted = action_vec_sel_joint[ind_sort]

    label_state_next_pred = "$x_{t+1}$ (prediction)"
    label_state_next_groundtruth = "$\hat{x}_{t+1}$ (groundtruth)"

    if cfg_one_step.name_state2visualize == "joint_angular_pos":
        name_state2visualize = "Joint angular position"
    else:
        name_state2visualize = cfg_one_step.name_state2visualize

    if cfg_one_step.ind_state_local == 8:
        ind_state_local = "Elbow 3"
    else:
        ind_state_local = cfg_one_step.ind_state_local

    freq_action = 10.
    time_vec = np.arange(len(state_groundtruth_sel_joint_sorted))/freq_action

    # pdb.set_trace()

    # Plotting stuff:
    if cfg_one_step.plotting:
        hdl_fig, hdl_plot = plt.subplots(1,1,figsize=(13,8))
        # hdl_fig.suptitle("Trained model roll-out; departing from a specific time step; using the true ations")
        hdl_fig.suptitle("Quality of next-state predictions: $x_{t+1} \sim p(\Delta x_{t+1} | \hat{x}_t, \hat{a}_t) + \hat{x}_t$")
        hdl_plot.plot(time_vec,absolute_state_np_predicted_sel_joint_sorted*scale_signal4plotting,label=label_state_next_pred,linestyle="--",color="lightcoral",marker=".")
        # hdl_plot.plot(absolute_state_np_predicted_sel_joint*scale_signal4plotting,label="state_next_predicted_unsorted",linestyle="--",color="green",marker=".")
        hdl_plot.plot(time_vec,state_groundtruth_sel_joint_sorted*scale_signal4plotting,linestyle="-",color="cornflowerblue",marker="None",linewidth=3.0,label=label_state_next_groundtruth)
        # hdl_plot.plot(delta_state_np_predicted[:,ind_state,0]*scale_signal4plotting,linestyle="-",color="lightcoral",marker="None",linewidth=1.5,label="delta_predicted_mean")
        hdl_plot.legend()
        # hdl_plot.set_title("Next state (sorted); {0:s} | index {1:d}".format(cfg_one_step.name_state2visualize,cfg_one_step.ind_state_local))
        hdl_plot.set_title("Next state (sorted) | {0:s} | {1:s}".format(name_state2visualize,ind_state_local))
        hdl_plot.set_ylabel(ylabel_state)
        hdl_plot.set_xlabel("time [sec]")
        hdl_plot.set_xlim([0,time_vec[-1]])
        hdl_plot.set_ylim([-120,-80])

        # hdl_plot[1].plot(action_vec_sel_joint_sorted*RAD2DEG,linestyle="-",color="brown",marker=".")
        # hdl_plot[1].set_title("Applied action (sorted); {0:s}".format(name_motors[ind_action]))
        # hdl_plot[1].plot(time_vec,action_vec_sel_joint*RAD2DEG,linestyle="-",color="brown",marker=".")
        # hdl_plot[1].set_title(r"Applied action $\hat{a}_t$ (unsorted)"+" | {0:s}".format(ind_state_local))
        # hdl_plot[1].set_xlabel("time [sec]")
        # hdl_plot[1].set_ylabel("State [deg]")
        # hdl_plot[1].set_xlim([0,time_vec[-1]])
        plt.show(block=cfg_one_step.block)

    return absolute_state_np_predicted_sel_joint_sorted, state_groundtruth_sel_joint_sorted

def generate_action_sequence(cfg_signal,ep_SAS_all):

    dt = 1./cfg_signal.freq_action_loop # Sampling period

    if cfg_signal.use == "linear":

        # Keep the same action in all the joints, and change the action of only one of the joints:
        step_init = time_index_local2global(cfg_signal.linear.step_init_local, cfg_signal.linear.which_traj, cfg_signal.linear.Nsteps)
        action = ep_SAS_all[step_init].a
        action_sequence = np.repeat(np.atleast_2d(action),cfg_signal.linear.Nsteps,axis=0)
        
        # Repeat the current state:
        state_curr = ep_SAS_all[step_init].s0
        state_curr_vec = np.repeat(np.atleast_2d(state_curr),cfg_signal.linear.Nsteps,axis=0)
        state_next_vec = state_curr_vec.copy()

        action_vec = np.linspace(   cfg_signal.linear.action_min,
                                    cfg_signal.linear.action_max,
                                    cfg_signal.linear.Nsteps)

        # Replace one of the actions:
        action_sequence[:,cfg_signal.which_joint] = action_vec

        # Compute corresponding time vector (for plotting):
        time_vec = np.arange(0,cfg_signal.linear.Nsteps)*dt

    elif cfg_signal.use == "sinewave":

        # Generate here the sine waves parameters:
        if cfg_signal.sinewave.freq_lims.sample:
            freq = np.random.uniform(low=cfg_signal.sinewave.freq_lims.low,high=cfg_signal.sinewave.freq_lims.high)
        else:
            freq = cfg_signal.sinewave.freq_lims.val

        if cfg_signal.sinewave.phase_lims.sample:
            phase = np.random.uniform(low=cfg_signal.sinewave.phase_lims.low,high=cfg_signal.sinewave.phase_lims.high)
        else:
            phase = cfg_signal.sinewave.phase_lims.val
            

        if cfg_signal.sinewave.ampl_lims.sample:
            ampl = np.random.uniform(low=cfg_signal.sinewave.ampl_lims.low,high=cfg_signal.sinewave.ampl_lims.high)
        else:
            ampl = cfg_signal.sinewave.ampl_lims.val
            
        print("freq:  {0:f} Hz".format(freq))
        print("phase: {0:f} deg".format(phase*RAD2DEG))
        print("ampl:  {0:f} deg".format(ampl*RAD2DEG))


        # Parameters:
        omega = 2.*np.pi*freq
        offset = 0.0

        # Time vector:
        t_max = 1./freq*cfg_signal.sinewave.repeat_wave # Repeat 5 times the entire wave
        Nchanges = int(round(t_max / dt))
        assert Nchanges > 2
        time_vec = np.linspace(0,t_max,Nchanges)
        action_vec = ampl*np.sin(omega*time_vec + phase) + offset
        # pdb.set_trace()

        # Keep the same action in all the joints, and change the action of only one of the joints:
        step_init = time_index_local2global(cfg_signal.sinewave.step_init_local,cfg_signal.sinewave.which_traj,Nchanges)
        action = ep_SAS_all[step_init].a
        action_sequence = np.repeat(np.atleast_2d(action),Nchanges,axis=0)
        
        # Repeat the current state:
        state_curr = ep_SAS_all[step_init].s0
        state_curr_vec = np.repeat(np.atleast_2d(state_curr),Nchanges,axis=0)
        state_next_vec = state_curr_vec.copy()

        # Replace one of the actions:
        action_sequence[:,cfg_signal.which_joint] = action_vec

    elif cfg_signal.use == "original":

        if cfg_signal.original.Nsteps == "None" or cfg_signal.original.Nsteps is None:
            Nsteps = len(ep_SAS_all)
        else:
            Nsteps = cfg_signal.original.Nsteps

        step_init = time_index_local2global(cfg_signal.original.step_init_local,cfg_signal.original.which_traj,Nsteps)
        step_end = step_init + Nsteps

        Nstates = len(ep_SAS_all[0].s0) # Actions: desired positions, [joints] (18)
        Nactions = len(ep_SAS_all[0].a) # State: [ joints , base position, base orientation ] (18 + 3 + 3)
        time_vec = dt*np.arange(step_init,step_end)

        # Get desired trajectory chunk as a numpy array:
        state_curr_vec = np.zeros((Nsteps,Nstates),dtype=np.float32)
        state_next_vec = np.zeros((Nsteps,Nstates))
        action_sequence = np.zeros((Nsteps,Nactions),dtype=np.float32)
        for k in range(Nsteps):
            state_curr_vec[k,:] = ep_SAS_all[k+step_init].s0.cpu().numpy()
            state_next_vec[k,:] = ep_SAS_all[k+step_init].s1.cpu().numpy()
            action_sequence[k,:] = ep_SAS_all[k+step_init].a.cpu().numpy()

    else:
        raise ValueError("which_action_signal_type: {linear,sinewave,original}")

    return action_sequence, state_curr_vec, state_next_vec, time_vec

def time_index_local2global(time_index_local,which_traj,Nsteps_episode):
    print("time_index_local: "+str(time_index_local)+" / "+str(Nsteps_episode))
    which_timestep = Nsteps_episode*which_traj + time_index_local
    print("which_timestep: "+str(which_timestep))
    return which_timestep 

def rollout_action_sequence(model, state0_k, actions_k, reward_func, propagator):

    assert actions_k.dim() == 3
    assert actions_k.shape[0] == 1

    # time_init = datetime.utcnow().timestamp()
    ret = propagator.compute_trajectories_and_returns(model, state0_k, actions_k, reward_func)
    # time_loop = print("Time elapsed: {0:2.2f} [ms]".format((datetime.utcnow().timestamp() - time_init)*1000))
    state0_k_ahead_plan_mean = ret.all_trajectories[0,:].mean(dim=0) # [1,Nparticles,Nplanning_horizon, Nstates] -> [Nparticles,Nplanning_horizon, Nstates] -> [Nplanning_horizon, Nstates]
    state0_k_ahead_plan_std = ret.all_trajectories[0,:].std(dim=0) # [1,Nparticles,Nplanning_horizon, Nstates] -> [Nparticles,Nplanning_horizon, Nstates] -> [Nplanning_horizon, Nstates]
    reward_k = ret.all_returns[0].item() # This is already averaged over the number of particles

    return state0_k_ahead_plan_mean, state0_k_ahead_plan_std, reward_k

# def plotting_predicted_actions_and_states(path,model_name,device,which_ind):
def plotting_predicted_actions_and_states(cfg_signal,path,analyze_model,my_model,device,ep_SAS_all_mean=None,ep_SAS_all_std=None):
    """

    TODO: Load here the planned action sequences
    """

    # step_init_local = cfg_signal.step_init_local
    # which_traj = cfg_signal.which_traj
    # Nsteps = cfg_signal.Nsteps

    # Select desired state index:
    ind_state, ind_action, scale_signal4plotting, ylabel_state = parse_user_specified_indices(cfg_signal=cfg_signal,Nstates=my_model.output_size)
    # pdb.set_trace()

    # Load data (unnormalized by default):
    ep_ind = load_data_online_pets(path,analyze_model,device,cfg_signal.which_traj)

    if ep_SAS_all_mean is not None and ep_SAS_all_std is not None:

        # Normalization
        how2normalize = "with_with_CURRENT_dataset_mean_and_var"
        # how2normalize = "with_with_TRAINING_dataset_mean_and_var"
        # how2normalize = ""
        if how2normalize == "with_with_TRAINING_dataset_mean_and_var": # Normalize data using the mean and std the NN was trained with
            
            # pdb.set_trace()
            Nsteps_as = len(ep_ind.planned_action_sequences)
            normalize_episodes(ep_ind.episode,ep_SAS_all_mean=ep_SAS_all_mean,ep_SAS_all_std=ep_SAS_all_std)
            for k in range(Nsteps_as):
                ep_ind.planned_action_sequences[k].actions_plan_mean = (ep_ind.planned_action_sequences[k].actions_plan_mean - ep_SAS_all_mean[1])/ep_SAS_all_std[1]
                ep_ind.planned_action_sequences[k].actions_plan_var = ep_ind.planned_action_sequences[k].actions_plan_var/(ep_SAS_all_std[1]**2)

        elif how2normalize == "with_with_TRAINING_dataset_mean_and_var":
            
            # # pdb.set_trace()
            sas_mean, sas_std = normalize_episodes(ep_ind.episode)
            Nsteps_as = len(ep_ind.planned_action_sequences)
            for k in range(Nsteps_as):
                ep_ind.planned_action_sequences[k].actions_plan_mean = (ep_ind.planned_action_sequences[k].actions_plan_mean - sas_mean[1])/sas_std[1]
                ep_ind.planned_action_sequences[k].actions_plan_var = ep_ind.planned_action_sequences[k].actions_plan_var/(sas_std[1]**2)

        else:
            log.info("Not normalizing")
    else:
        log.info("Not normalizing (!)")
    
    # [s0_mean,a_mean,s1_mean], [s0_std,a_std,s1_std]
    # pdb.set_trace()

    # Infer indices:
    Ntrajectories = ep_ind.planned_action_sequences[0].all_trajectories.shape[0]
    Nparticles = ep_ind.planned_action_sequences[0].all_trajectories.shape[1]
    Nplanning_horizon = ep_ind.planned_action_sequences[0].all_trajectories.shape[2] - 1
    Nstates = ep_ind.planned_action_sequences[0].all_trajectories.shape[3]
    Nactions = ep_ind.planned_action_sequences[0].actions[1]
    Nelites = ep_ind.planned_action_sequences[0].best_optimized_rewards.shape[0]
    Nsteps = cfg_signal.Nsteps
    # ind_state = cfg_signal.ind_state

    # assert cfg_signal.Nsteps <= len(ep_ind.planned_action_sequences), "cfg_signal.Nsteps={0:d} is larger than the episode length: {1:d}".format(cfg_signal.Nsteps,len(ep_ind.planned_action_sequences))
    # assert cfg_signal.Nsteps <= len(ep_ind.episode), "cfg_signal.Nsteps={0:d} is larger than the episode length: {1:d}".format(cfg_signal.Nsteps,len(ep_ind.episode))

    if cfg_signal.Nsteps is None:
        Nsteps = min(len(ep_ind.planned_action_sequences),len(ep_ind.episode))
        log.info("Using Nsteps = {0:d}".format(Nsteps))
    else:
        Nsteps = cfg_signal.Nsteps

    # # States
    # ep.all_trajectories # [Ntrajectories x Nparticles x Nplanning_horizon+1 x Nstates]
    # ep.states_plan_mean # [ Nplanning_horizon+1, Nstates ], with Nstates = len(state0)    # mean of the elites of all_trajectories
    # ep.states_plan_var # [ Nplanning_horizon+1, Nstates ], with Nstates = len(state0)     # var of the elites of all_trajectories

    # # Actions:
    # ep.actions_plan_mean # [Nplanning_horizon x action_dim] # mean of the elites of new actions
    # ep.actions_plan_var # [Nplanning_horizon x action_dim] # var of the elites of new actions
    # ep.actions # [Nplanning_horizon x action_dim] # -> Same as actions_plan_mean
    
    # # Rewards:
    # ep.all_optimized_rewards # [Ntrajectories,] -> Same as ep.all_returns
    # ep.best_optimized_rewards   # [Nelites,]
    
    # # Others:
    # ep.all_actions # -> None
    # ep.all_returns # -> Same as all_optimized_rewards
    # ep.seq_return # -> None

    Nparticles = 20
    which_prop = "ts_prop" # 12 ms / 100 particles (both, with and without permutations) -> 45 times faster
    # which_prop = "single_prop" # 550 ms / 100 particles
    if which_prop == "single_prop":
        propagator = SimpleProp(batch_multiplier=1, particles=Nparticles, return_trajectories=True, use_full_traj=False)
    elif which_prop == "ts_prop":
        propagator = TSProp(particles=Nparticles, return_trajectories=True, permute_assignment=True)
    else:
        raise ValueError("Nooorrr!!!!")

    # Create reward:
    # reward_func = mbrl.rewards.RewardWalkForward.get_reward_signal_static
    reward_class = mbrl.rewards.RewardWalkForward(dim_state=Nstates)
    reward_func = reward_class.get_reward_signal


    # Decide what to plot:
    plot_states_plan = False
    plot_trajs = False
    class_members = dir(ep_ind.planned_action_sequences[0])
    if "states_plan_mean" in class_members and "states_plan_var" in class_members:
        plot_states_plan = True
    elif ep_ind.planned_action_sequences[0].all_trajectories is not None:
        plot_trajs = True
    
    # Get predicted action sequence:
    pdb.set_trace()
    actions_planned_mean = np.zeros((Nplanning_horizon,Nsteps))
    actions_planned_std = np.zeros((Nplanning_horizon,Nsteps))
    states_planned_mean = np.zeros((Nplanning_horizon+1,Nsteps))
    states_planned_std = np.zeros((Nplanning_horizon+1,Nsteps))
    action_vec = np.zeros(Nsteps)
    state_vec = np.zeros(Nsteps)
    rewards_mean = np.zeros(Nsteps)
    rewards_std = np.zeros(Nsteps)
    # pdb.set_trace()
    for k in range(Nsteps):
        
        # Actions prediction:
        # pdb.set_trace()
        actions_planned_mean[:,k]   = ep_ind.planned_action_sequences[k].actions_plan_mean[:,ind_action].to("cpu")
        actions_planned_std[:,k]    = np.sqrt(ep_ind.planned_action_sequences[k].actions_plan_var[:,ind_action].to("cpu"))
        action_vec[k]               = ep_ind.episode[k].a[ind_action].to("cpu")

        # State predictions:
        state_vec[k]                = ep_ind.episode[k].s0[ind_state].cpu()
        if plot_states_plan:

            # Get predicted states and reward:
            state0_k = ep_ind.episode[k].s0
            action_plan_k = ep_ind.planned_action_sequences[k].actions_plan_mean
            action_plan_k = action_plan_k.unsqueeze(0) # Add zero dimension. This just makes action_plan_k a 3D tensor, for which the first dimension (number of trajectories) is equal to one

            # Normalize before rolling out:
            # pdb.set_trace()
            # state0_k = (state0_k-ep_SAS_all_mean[0])/ep_SAS_all_std[0]
            # pdb.set_trace()
            # action_plan_k = (action_plan_k-ep_SAS_all_mean[1][None,:].repeat(Nplanning_horizon,1).unsqueeze(0))/ep_SAS_all_std[1][None,:].repeat(Nplanning_horizon,1).unsqueeze(0)
            state0_k_ahead_plan_mean, state0_k_ahead_plan_std, reward_k = rollout_action_sequence(my_model,state0_k,action_plan_k,reward_func,propagator)

            # Unnormalize:
            # state0_k = state0_k*ep_SAS_all_std[0] + ep_SAS_all_mean[0]
            # action_plan_k = action_plan_k*ep_SAS_all_std[1][None,:].repeat(Nplanning_horizon,1).unsqueeze(0) + ep_SAS_all_mean[1][None,:].repeat(Nplanning_horizon,1).unsqueeze(0)
            
            # Unnormalize:
            # pdb.set_trace()
            # states_planned_mean[:,k] = state0_k_ahead_plan_mean[:,ind_state]*ep_SAS_all_std[0][ind_state].item() + ep_SAS_all_mean[0][ind_state].item()
            # states_planned_std[:,k]  = state0_k_ahead_plan_std[:,ind_state]*ep_SAS_all_std[1][ind_state].item() + ep_SAS_all_mean[1][ind_state].item()

            # Without normalization:
            states_planned_mean[:,k] = state0_k_ahead_plan_mean[:,ind_state].to("cpu")
            states_planned_std[:,k]  = state0_k_ahead_plan_std[:,ind_state].to("cpu")

            rewards_mean[k]   = reward_k
            # rewards_std[k]    = torch.std(ep_ind.planned_action_sequences[k].best_optimized_rewards).item()
            rewards_std[k]    = 0.0

        elif plot_trajs: # In the absence of the above, use all the trajectories, which is incorrect, as we only need the elites
            raise
            all_trajs = ep_ind.planned_action_sequences[k].all_trajectories[:,ind_state]    # [ Ntraj, Nparticles, planning_horizon+1 ]
            aux = all_trajs.view(-1,Nplanning_horizon+1,Nstates)                        # [ Ntraj*Nparticles, planning_horizon+1 ]
            states_planned_mean[:,k] = torch.mean(aux, dim=0)               # [ planning_horizon+1, Nstates ], with Nstates = len(state0)
            states_planned_std[:,k] = torch.std(aux, unbiased=False, dim=0) # [ planning_horizon+1, Nstates ], with Nstates = len(state0)


    # # Unnormalize the data:
    # # pdb.set_trace()
    # s0_mean = ep_SAS_all_mean[0][ind_state].item()
    # s0_std = ep_SAS_all_std[0][ind_state].item()
    # states_planned_mean = states_planned_mean*s0_std + s0_mean
    # state_vec = state_vec*s0_std + s0_mean

    # a_mean = ep_SAS_all_mean[1][ind_state].item()
    # a_std = ep_SAS_all_std[1][ind_state].item()
    # actions_planned_mean = actions_planned_mean*a_std + a_mean
    # action_vec = action_vec*a_std + a_mean


    freq = 10.
    dt = 1./freq
    time_vec_tot = np.linspace(0.,Nsteps*dt,Nsteps+1)
    time_vec_tot = time_vec_tot[0:-1]    
    time_vec = dict(tot=time_vec_tot, actions_local=np.arange(0,Nplanning_horizon)*dt, states_local=np.arange(0,Nplanning_horizon+1)*dt)

    hdl_fig, hdl_plot = plt.subplots(2,1,figsize=(16,9),sharex=True)


    # colors = colormap.get_cmap("summer")
    # colors_mat = np.zeros((Nsteps-Nplanning_horizon,3))
    # for k in range(Nsteps-Nplanning_horizon):
    #     colors_mat[k,:] = colors(k/(Nsteps-Nplanning_horizon))[0:3]

    colors = colormap.get_cmap("viridis")
    colors_mat = np.zeros((Nplanning_horizon+1,3))
    for k in range(Nplanning_horizon+1):
        colors_mat[k,:] = colors(k/(Nplanning_horizon+1))[0:3]


    # For presentation
    # ================

    # Attributes:
    color_pred = "darkgreen"
    color_signal = "brown"
    color_rew = "mediumpurple"
    markersize_pred = 4
    linewidth_pred = 2.0
    plot_every = Nplanning_horizon+1
    errorevery = 1

    # hdl_fig.suptitle("Generated action sequences and model roll-out using CEM | Planning horizon: 5 steps (0.5 sec)")
    hdl_fig.suptitle("Generated action sequences and model roll-out using PDDM | Planning horizon: 5 steps (0.5 sec)")
    label_action_pred = "Optimal action sequence"
    
    if ind_state == 8:
        ylabel_state = "Angle [deg]"
        title_state = "State (Elbow 3)"
    elif ind_state == 18:
        ylabel_state = "Position [m]"
        title_state = "State (position X)"
    else:
        title_state = "State index {0:d}".format(ind_state)

    # Plot actions:
    hdl_plot[0].plot(time_vec["tot"],action_vec*RAD2DEG,linestyle="--",marker="s",markersize=3,color=color_signal,label="Executed actions")
    for k in range(Nsteps-Nplanning_horizon):
        color_pred = colors_mat[k % (Nplanning_horizon+1),:]
        if k % plot_every == 0:
            actions_planned_mean_plus_std = (actions_planned_mean[:,k] + actions_planned_std[:,k])*RAD2DEG
            actions_planned_mean_minus_std = (actions_planned_mean[:,k] - actions_planned_std[:,k])*RAD2DEG
            if k == 0:
                label_action_pred_use = label_action_pred
            else:
                label_action_pred_use = None
            hdl_plot[0].plot(time_vec["actions_local"],actions_planned_mean[:,k]*RAD2DEG,color=color_pred,linewidth=linewidth_pred,label=label_action_pred_use)
            hdl_plot[0].fill_between(time_vec["actions_local"],actions_planned_mean_minus_std,actions_planned_mean_plus_std,color=color_pred,alpha=0.2)
            # hdl_plot[0].errorbar(time_vec["actions_local"],actions_planned_mean[:,k]*RAD2DEG,yerr=actions_planned_std[:,k]*RAD2DEG,linestyle="-",color=color_pred,
            #                     linewidth=linewidth_pred,marker=".",markerfacecolor=color_pred,markersize=markersize_pred,errorevery=errorevery,capsize=4)

            # hdl_plot[0].errorbar(time_vec["actions_local"],actions_planned_mean[:,k]*RAD2DEG,yerr=actions_planned_std[:,k]*RAD2DEG,linestyle="-",color=color_pred,
            #                     linewidth=linewidth_pred,marker=".",markerfacecolor=color_pred,markersize=markersize_pred,errorevery=errorevery,capsize=4)
        time_vec["actions_local"] += dt

    hdl_plot[0].set_ylabel("Angle [deg]")
    # hdl_plot[0].set_xticks([])
    hdl_plot[0].set_title("Action")
    hdl_plot[0].set_facecolor("whitesmoke")
    # hdl_plot[0].set_facecolor("lightsteelblue")
    hdl_plot[0].set_xlim([time_vec["tot"][0],time_vec["tot"][-1]])
    # hdl_plot[0].set_ylim([100,200])
    hdl_plot[0].legend()



    # Plot states:
    label_state_pred = "state predictions"
    hdl_plot[1].plot(time_vec["tot"],state_vec*scale_signal4plotting,linestyle="--",marker="s",markersize=3,color=color_signal,label="state observed")
    if plot_states_plan or plot_trajs:
        for k in range(Nsteps-Nplanning_horizon):
            color_pred = colors_mat[k % (Nplanning_horizon+1),:]
            if k % plot_every == 0:
                if k == 0:
                    label_state_pred_use = label_state_pred
                else:
                    label_state_pred_use = None
                states_planned_mean_plus_std = (states_planned_mean[:,k] + states_planned_std[:,k])*scale_signal4plotting
                states_planned_mean_minus_std = (states_planned_mean[:,k] - states_planned_std[:,k])*scale_signal4plotting
                hdl_plot[1].plot(time_vec["states_local"],states_planned_mean[:,k]*scale_signal4plotting,color=color_pred,linewidth=linewidth_pred,label=label_state_pred_use)
                hdl_plot[1].fill_between(time_vec["states_local"],states_planned_mean_minus_std,states_planned_mean_plus_std,color=color_pred,alpha=0.2)
                # hdl_plot[1].errorbar(time_vec["states_local"],states_planned_mean[:,k]*scale_signal4plotting,yerr=states_planned_std[:,k]*scale_signal4plotting,linestyle="-",color=color_pred,
                #                     linewidth=linewidth_pred,marker=".",markerfacecolor=color_pred,markersize=markersize_pred,errorevery=errorevery,capsize=4)
            time_vec["states_local"] += dt

    # hdl_plot[1].set_xticks([])
    hdl_plot[1].set_title(title_state)
    hdl_plot[1].set_ylabel(ylabel_state)
    hdl_plot[1].set_facecolor("whitesmoke")
    hdl_plot[1].set_xlim([time_vec["tot"][0],time_vec["tot"][-1]])
    hdl_plot[1].set_xlabel("time [sec]")
    # hdl_plot[1].set_ylim([-1.0,+1.0])
    # hdl_plot[1].set_ylim([-0.1,0.3])
    # hdl_plot[1].set_ylim([-200,0.0])
    # hdl_plot[1].set_ylim([100,200])
    hdl_plot[1].legend()

    # hdl_plot[1].set_facecolor("lightsteelblue")

    # # Plot rewards:
    # hdl_plot[2].errorbar(time_vec["tot"],rewards_mean,yerr=rewards_std,linestyle="-",color=color_rew,
    #                     linewidth=linewidth_pred,marker=".",markerfacecolor=color_rew,markersize=markersize_pred,errorevery=errorevery,capsize=4)
    # hdl_plot[2].set_title("Reward")
    # hdl_plot[2].set_xlabel("time [sec]")
    # hdl_plot[2].set_facecolor("whitesmoke")

    # ========================================================================================================================================================







    # Original
    # ================


    # # Attributes:
    # color_pred = "darkgreen"
    # color_signal = "brown"
    # color_rew = "mediumpurple"
    # markersize_pred = 4
    # linewidth_pred = 0.7
    # plot_every = Nplanning_horizon+1
    # errorevery = 1

    
    # # Plot actions:
    # hdl_plot[0].plot(time_vec["tot"],action_vec*RAD2DEG,linestyle="--",marker="s",markersize=3,color=color_signal)
    # for k in range(Nsteps-Nplanning_horizon):
    #     color_pred = colors_mat[k % (Nplanning_horizon+1),:]
    #     if k % plot_every == 0:
    #         hdl_plot[0].errorbar(time_vec["actions_local"],actions_planned_mean[:,k]*RAD2DEG,yerr=actions_planned_std[:,k]*RAD2DEG,linestyle="-",color=color_pred,
    #                             linewidth=linewidth_pred,marker=".",markerfacecolor=color_pred,markersize=markersize_pred,errorevery=errorevery,capsize=4)
    #     time_vec["actions_local"] += dt

    # hdl_plot[0].set_ylabel("Action [deg]")
    # # hdl_plot[0].set_xticks([])
    # hdl_plot[0].set_title("Joint {0:s}".format(name_motors[ind_action]))
    # hdl_plot[0].set_facecolor("whitesmoke")
    # # hdl_plot[0].set_facecolor("lightsteelblue")

    # # Plot states:
    # hdl_plot[1].plot(time_vec["tot"],state_vec*scale_signal4plotting,linestyle="--",marker="s",markersize=3,color=color_signal)
    # if plot_states_plan or plot_trajs:
    #     for k in range(Nsteps-Nplanning_horizon):
    #         color_pred = colors_mat[k % (Nplanning_horizon+1),:]
    #         if k % plot_every == 0:
    #             hdl_plot[1].errorbar(time_vec["states_local"],states_planned_mean[:,k]*scale_signal4plotting,yerr=states_planned_std[:,k]*scale_signal4plotting,linestyle="-",color=color_pred,
    #                                 linewidth=linewidth_pred,marker=".",markerfacecolor=color_pred,markersize=markersize_pred,errorevery=errorevery,capsize=4)
    #         time_vec["states_local"] += dt

    # # hdl_plot[1].set_xticks([])
    # hdl_plot[1].set_title("State index {0:d}".format(ind_state))
    # hdl_plot[1].set_ylabel(ylabel_state)
    # hdl_plot[1].set_facecolor("whitesmoke")
    # # hdl_plot[1].set_facecolor("lightsteelblue")

    # # Plot rewards:
    # hdl_plot[2].errorbar(time_vec["tot"],rewards_mean,yerr=rewards_std,linestyle="-",color=color_rew,
    #                     linewidth=linewidth_pred,marker=".",markerfacecolor=color_rew,markersize=markersize_pred,errorevery=errorevery,capsize=4)
    # hdl_plot[2].set_title("Reward")
    # hdl_plot[2].set_xlabel("time [sec]")
    # hdl_plot[2].set_facecolor("whitesmoke")

    # pdb.set_trace()
    plt.show(block=True)

    return

def plotting_rewards(cfg_signal,path,analyze_model):
    """

    TODO: This function isn't adequate for the overall structure: ADAPT!
    Didn't do it before because I was lacking from the episode from the robodev
    """

    
    # Exclude trajectories errorc checking:
    trajectories_list = get_trajectory_list(cfg_signal.traj_init,cfg_signal.traj_end,cfg_signal.exclude)

    path += analyze_model["name_folder"] + "/"

    # Put all the episodes together:
    Nrews = len(trajectories_list)
    reward_list = []
    reward_cum = np.zeros(Nrews)
    jj = 0
    Nsteps_max = 0
    for which_traj in trajectories_list:

        # Backwards compatibility:
        if cfg_signal.load_rewards_from_saved_episodes:

            path_full = "{0:s}/{1:s}".format(path,cfg_signal.auxiliar_episode_name)
            path_full = path_full.replace("xxx",str(which_traj))
            reward_vec = np.concatenate(torch.load(path_full,map_location=torch.device(device_global)).rewards)
            reward_list += [reward_vec]
        else:
            path_full = "{0:s}/{1:s}".format(path,cfg_signal.base_name)
            path_full = path_full.replace("xxx",str(which_traj))
            reward_list += [torch.load(path_full,map_location=torch.device(device_global))] # This is concatenating numpy arrays in a list
        
        # reward_cum[jj] = np.sum(reward_list[jj])
        reward_cum[jj] = np.mean(reward_list[jj])
        # reward_cum[jj] = reward_list[jj][-1]

        # Find maximum:
        if len(reward_list[jj]) > Nsteps_max:
            Nsteps_max = len(reward_list[jj])

        jj += 1

    if cfg_signal.Nsteps_max is not None:
        Nsteps_max = cfg_signal.Nsteps_max

    reward_steps_mean = np.zeros(Nsteps_max)
    reward_steps_std = np.zeros(Nsteps_max)

    for k in range(Nsteps_max):

        list_non_finished_episodes_rewards = []
        for rew_epi in reward_list:
            if k < len(rew_epi):
                list_non_finished_episodes_rewards += [rew_epi[k]]
        list_non_finished_episodes_rewards = np.array(list_non_finished_episodes_rewards)

        # Reward at each iteration:
        reward_steps_mean[k] = np.mean(list_non_finished_episodes_rewards)
        reward_steps_std[k] = np.std(list_non_finished_episodes_rewards)


    # ylims_rt = None
    # ylims_Rt = None

    # pdb.set_trace()
    # Good walking experimentis
    ylims_rt = [-103,-98]
    ylims_Rt = [-130,-90]

    # 100 experiments with CEM:
    # ylims_Rt = [-30,0]

    hdl_fig, hdl_plot = plt.subplots(1,1,figsize=(12,6),sharex=True)
    freq_action = 10 # Hz
    time_vec = (np.arange(Nsteps_max)+1)/freq_action
    # for rew_epi in reward_list:
    #     hdl_plot.plot(rew_epi,color="gray",linewidth=0.5)
    hdl_plot.plot(time_vec,reward_steps_mean,linestyle="-",color="mediumpurple",linewidth=2,label="mean")
    hdl_plot.fill_between(time_vec, reward_steps_mean - reward_steps_std, reward_steps_mean + reward_steps_std,color='mediumpurple', alpha=0.2,label="std")
    hdl_plot.set_xlabel("Task time [sec]")
    hdl_plot.set_ylabel(r"Inst. Reward $r_t$")
    hdl_plot.set_title("Instantaneous reward over episodes")
    hdl_plot.set_facecolor("whitesmoke")
    hdl_plot.set_ylim(ylims_rt)
    hdl_plot.set_xlim([0,time_vec[-1]])
    # hdl_plot.legend(loc="lower right")
    hdl_plot.legend(loc="upper left")


    hdl_fig, hdl_plot = plt.subplots(1,1,figsize=(12,6),sharex=True)
    Nrews_vec = np.arange(Nrews)+1
    hdl_plot.plot(Nrews_vec,reward_cum,linestyle="-",color="mediumpurple",linewidth=4)
    hdl_plot.set_xlabel("Number of episodes")
    hdl_plot.set_ylabel(r"Cum. Reward $R_t$")
    hdl_plot.set_title("Cumulative reward at each episode")
    hdl_plot.set_facecolor("whitesmoke")
    hdl_plot.set_ylim(ylims_Rt)
    hdl_plot.set_xlim([1,Nrews])
    if ylims_Rt is not None:
        hdl_plot.fill_between(Nrews_vec, ylims_Rt[0]*np.ones(Nrews), reward_cum,color='mediumpurple', alpha=0.2)


def predictions_triangular(cfg,path,model_pretrained,analyze_model):
    """

    path: base path, without including the walking experiments, i.e., cfg.data2load.cpg.path
    model_pretrained: The pre-trained dynamics model
    """

    # Re-trained models:
    name_model_retrained = cfg.predictions_triangular.name_model_retrained

    # Load the trained model:
    model_curr = model_pretrained

    # Max number of episodes:
    Nepisodes = cfg.predictions_triangular.Nepisodes
    
    # Pre-load data acquired over PETS episodes:
    path_with_walking = "{0:s}/{1:s}/".format(path,analyze_model["name_folder"])
    Nepisodes = cfg.predictions_triangular.Nepisodes
    ep_SAS_all = [None]*Nepisodes
    for ind_epi in range(Nepisodes):
        name_episode = cfg.predictions_triangular.name_episode
        name_episode = name_episode.replace("xxx",str(ind_epi))
        ep_SAS_all[ind_epi], _, _ = load_ep_SAS(path=path_with_walking,
                                                traj_init=ind_epi,traj_end=ind_epi+1,
                                                exclude_trajs=None,
                                                base_name2save="",
                                                name_episode=name_episode)

    # Call the one step ahead:
    RMSE_matrix = np.zeros((Nepisodes,Nepisodes))
    cfg.one_step_predictions_sorted.plotting = False
    cfg.one_step_predictions_sorted.original.step_init_local = 0
    for ind_epi_row in range(Nepisodes):

        # Update the model when available:
        if (ind_epi_row+1) % cfg.predictions_triangular.new_dynamics_model_every_steps == 0:
            name_model_retrained = name_model_retrained.replace("xxx",str(ind_epi_row))
            model_curr = load_dynamics_model(path,analyze_model["name_folder"],name_model_retrained,cfg.device_global)

        # Load all the subsequent data episodes, and see how the loaded model can predict them
        for ind_epi_col in range(ind_epi_row,Nepisodes):

            cfg.one_step_predictions_sorted.original.which_traj = 0
            cfg.one_step_predictions_sorted.original.Nsteps = len(ep_SAS_all[ind_epi_col])
            # pdb.set_trace()

            # Use the same model to compute the one step ahead predicitions of all the remaining episodes:
            absolute_state_np_predicted_sel_joint_sorted, \
            state_groundtruth_sel_joint_sorted = one_step_predictions_sorted(cfg.one_step_predictions_sorted,ep_SAS_all[ind_epi_col],model_curr)

            # Compute MSE:
            MSE = la.norm(absolute_state_np_predicted_sel_joint_sorted-state_groundtruth_sel_joint_sorted) 

            # RMSE_matrix[ind_epi_row,ind_epi_col] = MSE / len(ep_SAS_all[ind_epi_col])
            RMSE_matrix[ind_epi_row,ind_epi_col] = MSE

    RMSE_matrix = RMSE_matrix / RMSE_matrix.max()

    hdl_fig, hdl_plot = plt.subplots(1,1,figsize=(8,8))
    # neg = plt.imshow(RMSE_matrix,cmap="gnuplot",origin="upper") # Like this the plot is like the matrix
    neg = plt.imshow(np.rot90(RMSE_matrix,k=1),cmap="gnuplot",origin="upper") # Like this, is what we want
    hdl_fig.colorbar(neg, ax=hdl_plot)
    hdl_plot.set_xlabel("Episodes")
    hdl_plot.set_ylabel("Episodes")
    hdl_plot.set_title("RMSE")
    print(RMSE_matrix)
    print(np.rot90(RMSE_matrix,k=1))
    # pdb.set_trace()
    # hdl_plot.set_xlim([0,Nepisodes])
    # hdl_plot.set_ylim([0,Nepisodes])
    # for k in range(Nepisodes):
    #     hdl_plot.get_xticks()



def plotting_training_loss_evolution(cfg_signal,train_loss_vec,test_loss_vec):

    hdl_fig, hdl_plot = plt.subplots(2,1,figsize=FIGSIZE)
    hdl_plot[0].plot(-train_loss_vec,label="train_log.train_losses",linestyle="--",color="cornflowerblue",marker=".")
    hdl_plot[0].set_title("train_log.train_losses (Negative log probability)")
    hdl_plot[0].set_ylabel("-log(p)")
    
    hdl_plot[1].plot(-test_loss_vec,label="train_log.test_losses",linestyle="--",color="cornflowerblue",marker=".")
    hdl_plot[1].set_title("train_log.test_losses (Negative log probability)")
    hdl_plot[1].set_xlabel("Nr. epochs")
    hdl_plot[1].set_ylabel("-log(p)")
    plt.show(block=cfg_signal.block)

    return hdl_fig

def plot_evolution_init(figsize=(15,8),indices2monitor=1):

    Nstates2monitor = len(indices2monitor)
    hdl_fig, hdl_plot = plt.subplots(Nstates2monitor+1,1,figsize=figsize)

    hdl_plot[0].set_ylabel("Reward")
    for k in range(Nstates2monitor):

        if indices2monitor[k] == 18:
            ylabel = "X [m]"
        elif indices2monitor[k] == 20:
            ylabel = "Z [m]"
        elif indices2monitor[k] == 22:
            ylabel = "sin(phiX)"
        elif indices2monitor[k] == 24:
            ylabel = "sin(phiY)"

        hdl_plot[k+1].set_ylabel(ylabel)
        hdl_plot[k].set_xticks([])
    hdl_plot[Nstates2monitor].set_xlabel("time [sec]") # Show the X-axis only in the last (bottom) plot

    return hdl_fig, hdl_plot

def add_plot_evolution(hdl_plot,cfg,rewards_k,state_k,trial_num,indices2monitor,Ntrials,color_epi,block):

    # Compute times:
    time_vec_rew = np.arange(0,len(rewards_k))/cfg.env.freq_action
    time_vec_sta = np.arange(0,len(state_k))/cfg.env.freq_action

    hdl_plot[0].plot(time_vec_rew,rewards_k,label="Episode {0:d}".format(trial_num+1),color=color_epi)
    # hdl_plot[0].legend()

    for ii in range(len(indices2monitor)):
        if indices2monitor[ii] in range(18,21):
            local_scale = 1.0
        elif indices2monitor[ii] in range(21,27):
            local_scale = 1.0
        else:
            local_scale = RAD2DEG
        hdl_plot[ii+1].plot(time_vec_sta,state_k[:,ii]*local_scale,label="Episode {0:d}".format(trial_num+1),color=color_epi)
        # hdl_plot[ii+1].legend()

    # # Baseline:
    # hdl_plot[1].plot(np.array([time_vec_rew[0],time_vec_rew[-1]]),-80*np.ones(2),color="grey",linestyle="-",linewidth=1.0)

    plt.show(block=block)
    plt.pause(0.5)

    return hdl_plot

def save_plot_evolution(hdl_fig,path_model_analysis,file_name_global,trial_num):

    figname = "{0:s}_{1:d}_figure.png".format(file_name_global,trial_num)
    log.info("Saving figure...")
    hdl_fig.savefig("{0:s}/{1:s}".format(path_model_analysis,figname),dpi=300)
    log.info("Done!")
    return

def parse_user_specified_indices(cfg_signal,Nstates):

    ind_state = None
    try:
        indices_of = eval(cfg_signal.ind_state_vector_dict)
        ind_state = indices_of[cfg_signal.name_state2visualize][cfg_signal.ind_state_local]
    except Exception as inst:
        print(type(inst),inst.args,inst)
        print("ind_state: ",ind_state)
        pdb.set_trace()

    # Infer the length of the state vector from cfg_signal.ind_state_vector_dict specified in the yaml file.
    # For this, we look for the maximum existing index:
    ind_largest = 0
    for indices_names,indices_val in indices_of.items():
        if indices_val[-1] > ind_largest:
            ind_largest = indices_val[-1]

    # Sanity check:
    assert ind_largest+1 == Nstates,    "The number of states in the data doesn't agree with the number of states inferred from the indices dictionary.\n \
                                         Check the cfg_signal.ind_state_vector_dict is well defined"

    if cfg_signal.name_state2visualize in ["joint_angular_pos","joint_angular_vel"]:
        ind_action = cfg_signal.ind_state_local
        scale_signal4plotting = RAD2DEG
        ylabel_state = "Angle [deg]"
    else:
        ind_action = 0
        scale_signal4plotting = 1.0
        if cfg_signal.name_state2visualize == "base_position":
            ylabel_state = "Position [m]"

    return ind_state, ind_action, scale_signal4plotting, ylabel_state

# def plot_rewards(cfg,rewards_list,state_list,trial_num,path_model_analysis):

#     # Printing:
#     hdl_fig, hdl_plot = plt.subplots(2,1,figsize=(15,8))
#     for k in range(len(rewards_list)):
#         time_vec_rew = np.arange(0,len(rewards_list[k]))/cfg.env.freq_action
#         time_vec_sta = np.arange(0,len(state_list[k]))/cfg.env.freq_action
#         hdl_plot[0].plot(time_vec_rew,rewards_list[k],label="Episode {0:d}".format(k+1))
#         for ii in range(state_list[k].shape[1]):
#             hdl_plot[1].plot(time_vec_sta,state_list[k][:,ii]*RAD2DEG,label="Episode {0:d}".format(k+1))

#     hdl_plot[0].set_ylabel("Reward")
#     # hdl_plot[0].set_xlabel("time [sec]")
#     hdl_plot[0].legend()
#     hdl_plot[1].set_ylabel("State [deg]")
#     hdl_plot[1].set_xlabel("time [sec]")
#     hdl_plot[1].legend()
#     plt.show(block=False)

#     if cfg.analyze_model.save_plot_evolution:
#         figname = "{0:s}_fig_trial_num_{0:d}.png".format(file_name_global,trial_num)
#         # figname = "go2position_singlejoint{0:d}_less_move.png".format(which_joint)
#         hdl_fig.savefig("{0:s}_{1:s}".format(path_model_analysis,figname))

