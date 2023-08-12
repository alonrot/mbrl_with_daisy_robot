import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import scipy.stats as stats
import pdb
import matplotlib.cm as colormap
import scipy.linalg as la



def main():

    dim = 2 # action srpace dimension
    planning_horizon = 12 # steps
    Ntrajectories = 2
    Niters = 4

    lim_low_single = 3.*np.array([-1,-1])
    lim_high_single = 3.*np.array([+1,+1])

    # Replicate howizontally:
    lim_low = np.tile(lim_low_single,(1,planning_horizon))
    lim_high = np.tile(lim_high_single,(1,planning_horizon))

    mean_init, var_init = reset_trajectory(lim_low,lim_high,fac_var=256)
    # mean_init = 0.5*np.tile(np.linspace(0,4.0,planning_horizon)[:,None],(1,dim))
    # mean_init = mean_init.reshape(-1)

    do_cem_full(lim_low_single,lim_high_single,mean_init,var_init,Niters,Ntrajectories,planning_horizon,dim)

    mean_init, var_init = reset_trajectory(lim_low_single,lim_high_single,fac_var=256)
    mean_nominal = np.tile(mean_init,(planning_horizon,1))
    # mean_nominal = 0.5*np.tile(np.linspace(0,4.0,planning_horizon)[:,None],(1,dim))
    # pdb.set_trace()
    # mean_nominal = 0.5*np.tile(np.linspace(0,4.0,planning_horizon)[:,None],(1,dim))
    do_brownian_motion_full(lim_low_single,lim_high_single,mean_nominal,var_init,Niters,Ntrajectories,planning_horizon,dim)

    mean_init, var_init = reset_trajectory(lim_low,lim_high,fac_var=256)
    # mean_nominal = 0.5*np.tile(np.linspace(0,4.0,planning_horizon)[:,None],(1,dim))
    do_pddm_full(lim_low_single,lim_high_single,mean_nominal,var_init,Niters,Ntrajectories,planning_horizon,dim)


    plt.show(block=True)



# CEM
# ===

def do_cem_full(lim_low_single,lim_high_single,mean_init,var_init,Niters,Ntrajectories,planning_horizon,dim):

    trajectories_and_iters = np.zeros((Niters,Ntrajectories,planning_horizon,dim))
    ind_elite = np.array([0,1]) # Select one trajectory faking the elties selection
    for ii in range(Niters):
        trajectories_and_iters[ii,:] = do_cem_iteration(Ntrajectories,mean_init,var_init,planning_horizon,dim)
        mean_init = trajectories_and_iters[ii,ind_elite,:,:].mean(axis=0).reshape(-1)

        # # Do the padding:
        # mean_init[0:-1,:] = mean_init[1::,:]
        # mean_init[-1,:] =


    # plot_trajectories(trajectories,lim_low_single,lim_high_single,title="CEM iteration")
    plot_trajectories_alliters(trajectories_and_iters,lim_low_single,lim_high_single,title="CEM iteration")

    # plt.show(block=True)

def do_cem_iteration(Ntrajectories,mean_init,var_init,planning_horizon,dim):

    samples_raw = stats.truncnorm.rvs(-2.,+2.,size=(Ntrajectories,planning_horizon*dim))

    samples = mean_init + samples_raw*np.sqrt(var_init)

    trajectories = np.zeros((Ntrajectories,planning_horizon,dim))
    for k in range(Ntrajectories):
        trajectories[k,:,:] = np.reshape(samples[k,:],(planning_horizon,dim))

    return trajectories


# ===============
# Brownian motion
# ===============

def do_brownian_motion_full(lim_low_single,lim_high_single,mean_nominal,var_init,Niters,Ntrajectories,planning_horizon,dim):

    trajectories_and_iters = np.zeros((Niters,Ntrajectories,planning_horizon,dim))
    rewards = npr.uniform(low=-1.,high=0.,size=(Ntrajectories))
    # goal = 4*np.ones(dim)
    # rewards = -la.norm(np.tile(goal,(Ntrajectories,1)) - trajectories_and_iters[ii,:,-1,:],axis=1)
    for ii in range(Niters):
        trajectories_and_iters[ii,:] = do_brownian_motion(Ntrajectories,mean_nominal,var_init,planning_horizon,dim,lim_low_single,lim_high_single)
        mean_nominal = do_PI2_mean(trajectories_and_iters[ii,:,:,:],rewards,planning_horizon)

    # Do the padding:
    mean_nominal[0:-1,:] = mean_nominal[1::,:]
    mean_nominal[-1,:],_ = reset_trajectory(lim_low_single,lim_high_single)

    # plot_trajectories(trajectories,lim_low_single,lim_high_single,title="CEM iteration")
    plot_trajectories_alliters(trajectories_and_iters,lim_low_single,lim_high_single,title="Brownian motion iteration")

    # plt.show(block=True)


def do_brownian_motion(Ntrajectories,mean_nominal,var_init,planning_horizon,dim,lim_low_single,lim_high_single):

    # mean_single = mean_init[0,0:dim] # Center of the dfomain
    std_single = np.sqrt(var_init)
    print(std_single)
    
    n_samples = np.zeros((Ntrajectories,planning_horizon,dim))
    trajectories = np.zeros((Ntrajectories,planning_horizon,dim))

    ind_rejected = np.ones(Ntrajectories,dtype=bool)
    sample = np.zeros((Ntrajectories,dim))
    sample_prev = np.zeros((Ntrajectories,dim)) # not do fi=or fairness with the other implementations

    # pdb.set_trace()
    trajectories[:,0,:] = np.tile(mean_nominal[0,:],(Ntrajectories,1)) + std_single*npr.normal(size=(Ntrajectories,dim))
    # trajectories[:,0,:] = np.tile(mean_nominal[0,:],(Ntrajectories,1))
    for jj in range(planning_horizon-1):

        ind_rejected[:] = True
        sample[:] = 0.0
        while ind_rejected.sum() > 0:

            # sample = npr.multivariate_normal(mean=mean_zero,cov=cov)
            # pdb.set_trace()
            sample_subset = std_single*npr.normal(size=(ind_rejected.sum(),dim))
            sample[ind_rejected,:] = sample_subset

            ind_rejected[ind_rejected] = ~((sample[ind_rejected,:]*sample_prev[ind_rejected,:]).sum(axis=1) >= 0.)

        # pdb.set_trace()
        assert np.all((sample*sample_prev).sum(axis=1) >= 0.0)

        # pdb.set_trace()
        # n_samples[:,jj+1,:] = n_samples[:,jj,:] + sample
        sample_prev[:] = sample

        trajectories[:,jj+1,:] = trajectories[:,jj,:] + sample

    return trajectories


def do_PI2_mean(trajectories,rewards,planning_horizon):

    # pdb.set_trace()

    exp_rew = np.exp(rewards)
    mean_nominal = np.zeros((planning_horizon,trajectories.shape[2]))
    for jj in range(planning_horizon):
        mean_nominal[jj,:] = np.matmul(exp_rew,trajectories[:,jj,:])

    exp_rew_sum = exp_rew.sum()
    assert exp_rew_sum > 1e-6
    mean_nominal = mean_nominal / exp_rew_sum

    return mean_nominal



# ===============
# PDDM
# ===============

def do_pddm_full(lim_low_single,lim_high_single,mean_nominal,var_init,Niters,Ntrajectories,planning_horizon,dim):

    trajectories_and_iters = np.zeros((Niters,Ntrajectories,planning_horizon,dim))
    rewards = npr.uniform(low=-1.,high=0.,size=(Ntrajectories))
    for ii in range(Niters):
        trajectories_and_iters[ii,:] = do_pddm(Ntrajectories,mean_nominal,var_init,planning_horizon,dim,lim_low_single,lim_high_single)
        mean_nominal = do_PI2_mean(trajectories_and_iters[ii,:,:,:],rewards,planning_horizon)


    # Do the padding:
    mean_nominal[0:-1,:] = mean_nominal[1::,:]
    mean_nominal[-1,:],_ = reset_trajectory(lim_low_single,lim_high_single)



    # plot_trajectories(trajectories,lim_low_single,lim_high_single,title="CEM iteration")
    plot_trajectories_alliters(trajectories_and_iters,lim_low_single,lim_high_single,title="PDDM iteration")

    # plt.show(block=True)



def do_pddm(Ntrajectories,mean_nominal,var_init,planning_horizon,dim,lim_low_single,lim_high_single):

    # mean_single = mean_init[0,0:dim] # Center of the dfomain
    var_single = var_init[0,0:dim] # Intiial variance
    std_single = np.sqrt(var_single)

    beta = 0.9

    # mean_nominal = np.repeat(mean_single[None,:],planning_horizon,axis=0)
    # trajectories = np.zeros((Ntrajectories,planning_horizon,dim))
    # pdb.set_trace()

    # Sample the elements of the random walk:
    u_samples = stats.truncnorm.rvs(-2.,+2.,loc=0,scale=std_single,size=(Ntrajectories,planning_horizon+1,dim))
    n_samples = np.zeros((Ntrajectories,planning_horizon+1,dim))
    for t in range(planning_horizon):
        n_samples[:,t+1,:] = beta*u_samples[:,t+1,:] + (1.-beta)*n_samples[:,t,:]

    # pdb.set_trace()

    trajectories = mean_nominal + n_samples[:,1:planning_horizon+1,:] # We only add the n_samples after the index 0, since those are all zero (initialization as in the paper)

    return trajectories


def plot_trajectories(trajectories,lim_low_single,lim_high_single,figsize=(8,8),title=""):

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

    # hdl_plot.legend()
    lims = np.vstack((lim_low_single,lim_high_single))
    hdl_plot.set_xlim(lims[:,0])
    hdl_plot.set_ylim(lims[:,1])
    hdl_plot.set_xlabel("dim 1")
    hdl_plot.set_ylabel("dim 2")

def plot_trajectories_alliters(trajectories_and_iters,lim_low_single,lim_high_single,figsize=(8,8),title=""):

    Niters = trajectories_and_iters.shape[0]
    Ntrajectories = trajectories_and_iters.shape[1]
    planning_horizon = trajectories_and_iters.shape[2]
    dim = trajectories_and_iters.shape[3]
    assert dim == 2, "Function not suited for dim > 2"

    colors = colormap.get_cmap("nipy_spectral")
    colors_mat = np.zeros((Ntrajectories,3))
    for k in range(Ntrajectories):
        colors_mat[k,:] = colors((k % Ntrajectories)/Ntrajectories)[0:3]

    assert Niters == 4
    hdl_fig, hdl_plot = plt.subplots(2,2,figsize=figsize)
    hdl_fig.suptitle(title)

    cc = 0
    for ii in range(Niters):

        if ii == 2:
            cc += 1

        # Plot trajectories:
        for k in range(Ntrajectories):
            hdl_plot[cc,ii%2].plot(trajectories_and_iters[ii,k,:,0],trajectories_and_iters[ii,k,:,1],color=colors_mat[k,:],label="traj{0:d} | iter{1:d}".format(k,ii),marker="s",linestyle="--")

            # Plot initial point:
            hdl_plot[cc,ii%2].plot(trajectories_and_iters[ii,k,0,0],trajectories_and_iters[ii,k,0,1],color="red",marker="*",linestyle="--",markersize=10)

        # hdl_plot[cc,ii%2].legend()
        lims = np.vstack((lim_low_single,lim_high_single))
        # hdl_plot[cc,ii%2].set_xlim([-4.,+4])
        # hdl_plot[cc,ii%2].set_ylim([-4.,+4])
        # hdl_plot[cc,ii%2].set_xlim(lims[:,0])
        # hdl_plot[cc,ii%2].set_ylim(lims[:,1])
        hdl_plot[cc,ii%2].set_xlabel("dim 1")
        hdl_plot[cc,ii%2].set_ylabel("dim 2")


def reset_trajectory(lim_low,lim_high,fac_var=16):

    mid_bound = (lim_low + lim_high) / 2
    mean_init = mid_bound

    # TODO: try to eliminate 16 here by using a better var_alpha
    # var = torch.pow(self.lower_bound - self.upper_bound, 2) / 16
    var = (lim_low - lim_high)**2 / fac_var
    var_init = var

    return mean_init, var_init



# Try the three of them


if __name__ == "__main__":
    main()