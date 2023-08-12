import numpy as np
import torch
import pdb
import hydra
import sys
from plotting_and_analysis.plotting_library import *
from tools.tools import load_data4analysis_and_training, fix_pose_for_model_training, remove_unwanted_states_func, load_dynamics_model, normalize_episodes, load_data_online_pets
from mbrl.utils import split_dataset, split_first_shuffle_after
import matplotlib.pyplot as plt

from scipy.fftpack import fft, fftfreq

@hydra.main(config_path='plotting_and_analysis/conf_plotting.yaml', strict=False)
def main(cfg):

    # Load and parse:
    data_and_opts = load_data4analysis_and_training(cfg)
    traj_init = data_and_opts["traj_init"]
    traj_end = data_and_opts["traj_end"]
    exclude_trajs = data_and_opts["exclude_trajs"]
    ind_selector = data_and_opts["ind_selector"]
    path_local = data_and_opts["path_local"]
    model_name = data_and_opts["model_name"]
    remove_unwanted_states = data_and_opts["remove_unwanted_states"]
    path = data_and_opts["path"]
    folder2model = data_and_opts["folder2model"]
    ep_SAS_all = data_and_opts["ep_SAS_all"]
    Nepi = data_and_opts["Nepi"]
    Nsteps_episode = data_and_opts["Nsteps_episode"]
    normalize_data = data_and_opts["normalize_data"]
    analyze_model = data_and_opts["analyze_model"]


    # Get all actions:
    Nel = len(ep_SAS_all)
    Nactions = len(ep_SAS_all[0].a)
    actions_all = np.zeros((Nel,Nactions))
    for k in range(Nel):
        actions_all[k,:] = ep_SAS_all[k].a
        # actions_all[k,:] = ep_SAS_all[k].s0[0:18]

    # pdb.set_trace()

    # Get main frequency component:
    freq_acq = 10 # Hz
    time_tot = Nel*(1./freq_acq)
    time_vec = np.linspace(0,time_tot,Nel)
    

    # Try the convolution:
    ind_pair = [0,12]
    ind_traj = np.arange(400,430)
    # conv_vec = np.convolve(actions_all[ind_traj,ind_pair[0]],actions_all[ind_traj,ind_pair[1]],mode="same")
    corr_vec = np.correlate(actions_all[ind_traj,ind_pair[0]],actions_all[ind_traj,ind_pair[1]],mode="same")
    xxx = np.arange(1-(ind_traj[-1]-ind_traj[0]),ind_traj[-1]-ind_traj[0])
    step_shift = xxx[corr_vec.argmax()]
    print(step_shift)
    pdb.set_trace()
    hdl_fig, hdl_plot = plt.subplots(3,1,figsize=(8,8))
    hdl_plot[0].plot(time_vec[ind_traj],actions_all[ind_traj,ind_pair[0]],label="a0")
    hdl_plot[1].plot(time_vec[ind_traj],actions_all[ind_traj,ind_pair[1]],label="a1")
    hdl_plot[2].plot(conv_vec,label="conv")
    plt.show(block=True)









    freq_peaks = np.zeros(Nactions)
    for k in range(Nactions):
        yf = fft(actions_all[:,k])
        # xf = fftfreq(Nel, 1./freq_acq)
        xf = np.linspace(0.0, 1.0/(2.0*(1./freq_acq)), Nel//2)
        freq_spectrum = 2.0/Nel * np.abs(yf[0:Nel//2])
        ind_max = np.argmax(freq_spectrum[1::])
        freq_peaks[k] = xf[ind_max]

        # hdl_fig, hdl_plot = plt.subplots(1,1,figsize=(8,8))
        # hdl_plot.plot(xf,freq_spectrum)
        # plt.show(block=True)
        # # pdb.set_trace()

    # hdl_fig, hdl_plot = plt.subplots(1,1,figsize=(8,8))
    # hdl_plot.plot(time_vec,actions_all[:,0])
    # plt.show(block=True)
    
    # Reshape by joints:
    freq_peaks_per_joint = freq_peaks.reshape((6,3)).mean(axis=0)
    print("freq_peaks_per_joint:",freq_peaks_per_joint)

    # frequency of acquisition/action:

    action_cycle_list = [None]*Nactions
    Ncycles_each_action = np.zeros(Nactions)
    Nsteps_each_cycle = np.zeros(Nactions,dtype=np.int)
    for k in range(Nactions):
        Ncycles_each_action[k] = time_tot/(1./freq_peaks[k])
        Nsteps_each_cycle[k] = int(np.round(Nel/Ncycles_each_action[k]))+1
        # action_cycle_list[k] = actions_all[0:-5,k].reshape((Nsteps_each_cycle[k],-1)).mean(axis=1)
        action_cycle_list[k] = actions_all[0:-5,k].reshape((-1,Nsteps_each_cycle[k]))


    hdl_fig, hdl_plot = plt.subplots(1,1,figsize=(8,8))
    for k in range(action_cycle_list[0].shape[0]):
        hdl_plot.plot(action_cycle_list[0][k,:])
    plt.show(block=True)


if __name__ == "__main__":
    sys.exit(main())




