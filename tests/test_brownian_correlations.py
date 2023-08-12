import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as stats
import pdb
import matplotlib.cm as colormap
import scipy.signal as sig


def main():

    dim = 2 # action srpace dimension
    planning_horizon = 20 # steps
    Ntrajectories = 4

    lim_low_single = np.array([-1,-1])
    lim_high_single = np.array([+1,+1])

    # Replicate howizontally:
    lim_low = np.tile(lim_low_single,(1,planning_horizon))
    lim_high = np.tile(lim_high_single,(1,planning_horizon))

    mean_init, var_init = reset_trajectory(lim_low,lim_high,fac_var=64)

    corr_matrices = get_prior_data(plotting=True)

    trajectories = do_brownian_motion_with_corre(Ntrajectories,mean_init,var_init,planning_horizon,dim,lim_low_single,lim_high_single,corr_matrices)
    plot_trajectories(trajectories,lim_low_single,lim_high_single,title="Directed brownian motion")

    plt.show(block=True)


def get_prior_data(figsize=(10,10),plotting=True):

    dim = 2
    assert dim == 2
    freq_action = 20
    freq = 1.
    ampl = np.array([[1.,2.]])
    offset = np.array([[-1.,0.5]])
    phase = np.array([[0.,np.pi/2]])
    sigma_n = 0.01

    time_tot = 5
    Nsteps = int(freq_action*time_tot)
    time_vec = np.linspace(0.0,time_tot,Nsteps)

    phase_vec = np.tile(phase.T,(1,Nsteps))
    ampl_vec = np.tile(ampl.T,(1,Nsteps))
    offset_vec = np.tile(offset.T,(1,Nsteps))
    time_vec = np.tile(time_vec[None,:],(2,1))

    # pdb.set_trace()
    sinewaves_data = offset_vec + ampl_vec*np.cos(2.*np.pi*freq*time_vec + phase_vec) + sigma_n*npr.normal(size=(2,Nsteps))

    # Pile up the waves and compute mean and variance:
    sinewaves_data_mean = sinewaves_data.reshape((2,time_tot,freq_action)).mean(axis=1)
    sinewaves_data_std = sinewaves_data.reshape((2,time_tot,freq_action)).std(axis=1)

    corr_vec = sig.correlate(sinewaves_data_mean[1,:],sinewaves_data_mean[0,:],mode="same")

    # Get deltas
    # pdb.set_trace()
    sinewaves_data_mean_deltas = np.diff(sinewaves_data_mean)
    Ndeltas = sinewaves_data_mean_deltas.shape[1]
    # Nperiod = sinewaves_data_mean.shape[1]
    corr_matrices = np.zeros((dim,dim,Ndeltas))
    for k in range(Ndeltas):

        theta1 = np.arctan2(sinewaves_data_mean_deltas[0,k],1./freq_action)
        theta2 = np.arctan2(sinewaves_data_mean_deltas[1,k],1./freq_action)
        
        corr_matrices[:,:,k] = np.eye(dim)
        corr_matrices[0,1,k] = np.cos(theta1 - theta2)
        corr_matrices[1,0,k] = corr_matrices[0,1,k]

        # # Scale:
        # corr_matrices[0,1,k] *= abs(sinewaves_data_mean_deltas[0,k]*sinewaves_data_mean_deltas[1,k])
        # corr_matrices[1,0,k] = corr_matrices[0,1,k]
        # corr_matrices[0,0,k] *= sinewaves_data_mean_deltas[0,k]**2
        # corr_matrices[1,1,k] *= sinewaves_data_mean_deltas[1,k]**2


        print(corr_matrices[:,:,k])

    if plotting:
        hdl_fig, hdl_plot = plt.subplots(2,1,figsize=figsize)
        hdl_plot[0].plot(time_vec[0,:],sinewaves_data[0,:],marker="s",linestyle="--")
        hdl_plot[1].plot(time_vec[0,:],sinewaves_data[1,:],marker="s",linestyle="--")
        # plt.show(block=True)

        hdl_fig, hdl_plot = plt.subplots(1,1,figsize=figsize)
        hdl_plot.plot(corr_matrices[1,0,:],marker="s",linestyle="--")
        hdl_plot.plot(corr_vec,marker="s",linestyle="--")

        hdl_fig, hdl_plot = plt.subplots(1,1,figsize=figsize)
        hdl_plot.plot(sinewaves_data[0,:],sinewaves_data[1,:],marker="s",linestyle="--")

        hdl_fig, hdl_plot = plt.subplots(2,1,figsize=figsize)
        hdl_plot[0].plot(sinewaves_data_mean[0,:],marker="s",linestyle="--")
        hdl_plot[1].plot(sinewaves_data_mean[1,:],marker="s",linestyle="--")

        # plt.show(block=True)

    return corr_matrices





# Generate here the matrices using the conovlution() corre() methids from scipy.stats, and see if they're the same as my method.
# If they're the same and faster, use them. If not, use my method.
# If this works, take the action sequence data, precompute the matrices, and see if we can replicate
# MAIN CRITICISM: This is almost like telling the robot waht to do.
# ADVANTAGE: It could be much much faster than BO
# We want one goal: move towarnds goal X. That could be one primitive.
# We want another goal: move towards Y. That could be another primitive.
# Same model... But what do we want the mode for, then? For example, what?

# The advantage of PETS is the supposed lack of prior structure...






def do_brownian_motion_with_corre(Ntrajectories,mean_init,var_init,planning_horizon,dim,lim_low_single,lim_high_single,corr_matrices):

    mean_single = mean_init[0,0:dim] # Center of the dfomain
    var_single = var_init[0,0:dim] # Intiial variance
    std_single = np.sqrt(var_single)
    
    planning_horizon = 30
    trajectories = np.zeros((Ntrajectories,planning_horizon,dim))

    # Initialization: The middle of the domain, for now
    for k in range(Ntrajectories):
        # pdb.set_trace()
        trajectories[k,0,:] = stats.truncnorm.rvs(-2.,+2.,loc=0.0,scale=std_single,)
        trajectories[k,1,:] = trajectories[k,0,:] + stats.truncnorm.rvs(-2.,+2.,loc=0.0,scale=std_single,)

    Ndeltas = corr_matrices.shape[2]

    for k in range(0,Ntrajectories):
        for j in range(1,planning_horizon-1):
        # for j in range(0,planning_horizon):

            ind_matrix = (j-1) % Ndeltas
            # corr_matrices[:,:,ind_matrix] = np.eye(2)
            # corr_matrices[0,1,ind_matrix] = -0.75
            # corr_matrices[1,0,ind_matrix] = -0.75
            # new_sample = npr.multivariate_normal(mean=np.zeros(dim),cov=corr_matrices[:,:,ind_matrix])

            reject = True
            while reject:
                vec_dic = trajectories[k,j,:] - trajectories[k,j-1,:]
                new_sample = npr.multivariate_normal(mean=np.zeros(dim),cov=corr_matrices[:,:,ind_matrix])
                reject = np.dot(vec_dic,new_sample) < 0. or np.any(trajectories[k,j,:] + new_sample < lim_low_single) or np.any(trajectories[k,j,:] + new_sample > lim_high_single)

            # pdb.set_trace()
            trajectories[k,j+1,:] = trajectories[k,j,:] + new_sample

    return trajectories





def plot_trajectories(trajectories,lim_low_single,lim_high_single,figsize=(10,10),title=""):

    Ntrajectories = trajectories.shape[0]
    planning_horizon = trajectories.shape[1]
    dim = trajectories.shape[2]
    assert dim == 2, "Function not suited for dim > 2"

    colors = colormap.get_cmap("nipy_spectral")
    Ncolors = 6
    colors_mat = np.zeros((Ntrajectories,3))
    for k in range(Ntrajectories):
        colors_mat[k,:] = colors((k % Ncolors)/Ncolors)[0:3]

    hdl_fig, hdl_plot = plt.subplots(1,1,figsize=figsize)
    hdl_fig.suptitle(title)

    # Plot trajectories:
    for k in range(Ntrajectories):
        hdl_plot.plot(trajectories[k,:,0],trajectories[k,:,1],color=colors_mat[k,:],label="traj1",marker="s",linestyle="--")

        # Plot initial point:
        hdl_plot.plot(trajectories[k,0,0],trajectories[k,0,1],color="red",marker="*",linestyle="--",markersize=10)

    hdl_plot.legend()
    lims = np.vstack((lim_low_single,lim_high_single))
    # hdl_plot.set_xlim(lims[:,0])
    # hdl_plot.set_ylim(lims[:,1])
    hdl_plot.set_xlabel("dim 1")
    hdl_plot.set_ylabel("dim 2")

    hdl_fig, hdl_plot = plt.subplots(2,1,figsize=figsize)
    for k in range(Ntrajectories):
        hdl_plot[0].plot(trajectories[k,:,0],marker="s",linestyle="--")
        hdl_plot[1].plot(trajectories[k,:,1],marker="s",linestyle="--")



def do_brownian_motion(Ntrajectories,mean_init,var_init,planning_horizon,dim,lim_low_single,lim_high_single):

    mean_single = mean_init[0,0:dim] # Center of the dfomain
    var_single = var_init[0,0:dim] # Intiial variance
    std_single = np.sqrt(var_single)
    
    trajectories = np.zeros((Ntrajectories,planning_horizon,dim))

    # Initialization: The middle of the domain, for now
    for k in range(Ntrajectories):
        # pdb.set_trace()
        trajectories[k,0,:] = stats.truncnorm.rvs(-2.,+2.,loc=0.0,scale=std_single,)
        trajectories[k,1,:] = trajectories[k,0,:] + stats.truncnorm.rvs(-2.,+2.,loc=0.0,scale=std_single,)

    for k in range(0,Ntrajectories):
        for j in range(1,planning_horizon-1):

            reject = True
            while reject:
                vec_dic = trajectories[k,j,:] - trajectories[k,j-1,:]
                new_sample = stats.truncnorm.rvs(-2.,+2.,loc=0.0,scale=std_single,)
                reject = np.dot(vec_dic,new_sample) < 0. or np.any(trajectories[k,j,:] + new_sample < lim_low_single) or np.any(trajectories[k,j,:] + new_sample > lim_high_single)

            # pdb.set_trace()
            trajectories[k,j+1,:] = trajectories[k,j,:] + new_sample

    return trajectories

def reset_trajectory(lim_low,lim_high,fac_var=16):

    mid_bound = (lim_low + lim_high) / 2
    mean_init = mid_bound

    # TODO: try to eliminate 16 here by using a better var_alpha
    # var = torch.pow(self.lower_bound - self.upper_bound, 2) / 16
    var = (lim_low - lim_high)**2 / fac_var
    var_init = var

    return mean_init, var_init



if __name__ == "__main__":
    main()