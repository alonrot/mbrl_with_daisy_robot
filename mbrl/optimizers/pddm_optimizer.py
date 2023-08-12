import logging
import numpy as np
import torch

from mbrl import utils
from .optimizer import Optimizer, ActionSequence

from datetime import datetime

import pdb

log = logging.getLogger(__name__)





class PDDM:
    """ Deep Dynamics Models for Learning Dexterous Maniulation
        Anusha Nagabandi, Kurt Konoglie, Sergey Levine, Vikash Kumar
        Google Brain
    """
    def __init__(self,
                 planning_horizon,
                 dim_action,
                 lower_bound,
                 upper_bound,
                 num_trajectories,
                 use_final_nominal_mean,
                 use_correlation_matrix,
                 max_iters,
                 num_elites):
        self.planning_horizon = planning_horizon
        self.dim_action = dim_action
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_trajectories = num_trajectories
        self.num_elites = num_elites
        self.max_iters = max_iters
        self.device = self.lower_bound.device

        # Added:
        self.use_final_nominal_mean = use_final_nominal_mean

        self.do_saturate = True

        self.saturations_count = [0,0]

        # self.result_dict = dict(mean_actions_optimized_elites=torch.zeros(self.planning_horizon, self.dim_action).to(device=self.device),
        #             actions_optimized_elites_var=torch.zeros(self.planning_horizon, self.dim_action).to(device=self.device), 
        #             best_values_index=None, 
        #             best_optimized_rewards=None, 
        #             all_optimized_rewards=torch.zeros(self.num_trajectories),
        #             saturations_count=[0,0])

        # import pdb; pdb.set_trace()

        self.gamma = 0.001
        self.beta = 0.6
        self.mean_nominal = torch.zeros(self.planning_horizon, self.dim_action).to(device=self.device)
        self.rewards = torch.zeros(self.planning_horizon, self.dim_action).to(device=self.device)

        if self.do_saturate:
            self.upper_bound_all = self.upper_bound.repeat((self.num_trajectories,self.planning_horizon,1))
            self.lower_bound_all = self.lower_bound.repeat((self.num_trajectories,self.planning_horizon,1))

        self.action_samples_ii = torch.zeros((self.num_trajectories,self.planning_horizon,self.dim_action)).to(device=self.device)
        self.dim_total = int(self.num_trajectories * self.planning_horizon * self.dim_action)

        self.u_samples = torch.zeros((self.num_trajectories,self.planning_horizon+1,self.dim_action)).to(device=self.device)
        self.n_samples = torch.zeros((self.num_trajectories,self.planning_horizon+1,self.dim_action)).to(device=self.device)

        # import pdb; pdb.set_trace()

        self.ind_rejected = torch.ones(self.num_trajectories,dtype=torch.int64).to(device=self.device)

        self.use_correlation_matrix = use_correlation_matrix
        if self.use_correlation_matrix:
            self.corr_matrix_chol = self.precompute_cholesky_matrix_for_sampling()


    def precompute_cholesky_matrix_for_sampling(self):

        # Useful indices combinations (used in motion_library.py)
        n_joints = 18
        ind_daisy_base = np.arange(0,n_joints,3)        # array([ 0,  3,  6,  9, 12, 15])
        ind_daisy_shoulder = np.arange(1,n_joints,3)    # array([ 1,  4,  7, 10, 13, 16])
        ind_daisy_elbow = np.arange(2,n_joints,3)       # array([ 2,  5,  8, 11, 14, 17])

        cov_fac = 0.98
        corr_matrix = torch.eye(n_joints).to(device=self.device)

        # Bases 1,4,5:
        corr_matrix[ind_daisy_base[0],ind_daisy_base[3]] = -cov_fac
        corr_matrix[ind_daisy_base[0],ind_daisy_base[4]] = +cov_fac
        corr_matrix[ind_daisy_base[3],ind_daisy_base[4]] = -cov_fac


        # Bases 2,3,6:
        corr_matrix[ind_daisy_base[1],ind_daisy_base[2]] = -cov_fac
        corr_matrix[ind_daisy_base[1],ind_daisy_base[5]] = +cov_fac
        corr_matrix[ind_daisy_base[2],ind_daisy_base[5]] = -cov_fac


        # Shoulders 1,4,5:
        corr_matrix[ind_daisy_shoulder[0],ind_daisy_shoulder[3]] = -cov_fac
        corr_matrix[ind_daisy_shoulder[0],ind_daisy_shoulder[4]] = +cov_fac
        corr_matrix[ind_daisy_shoulder[3],ind_daisy_shoulder[4]] = -cov_fac

        # Shoulders 2,3,6:
        corr_matrix[ind_daisy_shoulder[1],ind_daisy_shoulder[2]] = -cov_fac
        corr_matrix[ind_daisy_shoulder[1],ind_daisy_shoulder[5]] = +cov_fac
        corr_matrix[ind_daisy_shoulder[2],ind_daisy_shoulder[5]] = -cov_fac

        # See https://math.stackexchange.com/questions/2079137/generating-multivariate-normal-samples-why-cholesky to see why we want it to be upper triangular
        # We need it because we pre-multiply by the samples. If we mostmultiplied, we'd need it to be lower triangular
        # Note that U = L^T and U^T = L
        corr_matrix = (corr_matrix + corr_matrix.transpose(0,1)) - torch.eye(n_joints).to(device=self.device)

        return corr_matrix.cholesky(upper=True)

    def update_nominal_mean(self):
        """

        PI2 update, in principle only used for PDDM
        """

        # Compute the nominal mean by using the PI2-type weighting average:
        # print("[Check gamma] @update_nominal_mean | max(rewards): " + str(self.rewards.max()) + " min(rewards): " + str(self.rewards.min()) )
        exp_rew = torch.exp(self.gamma*self.rewards)
        if torch.any(exp_rew == 0.0):
            import pdb; pdb.set_trace()

        for t in range(self.planning_horizon):
            self.mean_nominal[t,:] = torch.matmul(exp_rew,self.action_samples_ii[:,t,:])

        exp_rew_sum = exp_rew.sum()
        assert exp_rew_sum > 1e-6, "Avoiding division by numerical zero. Probably gamma needs to be reduced..."
        self.mean_nominal = self.mean_nominal / exp_rew_sum


    def optimize(self, function, action_samples_init, std_action_samples, minimize=True):

        # Error checking:
        assert torch.is_tensor(action_samples_init)
        assert torch.is_tensor(std_action_samples)
        assert self.lower_bound.shape[0] == self.dim_action
        assert self.upper_bound.shape[0] == self.dim_action
        
        # Reinitialize:
        self.mean_nominal[:] = 0.0
        self.rewards[:] = 0.0
        self.action_samples_ii[:] = 0.0

        ii = 0
        cc_dbg = 0
        while ii < self.max_iters:

            # # Nominal mean:
            # if ii == 0:
            #     self.mean_nominal = action_samples_init
            # else:
            #     self.update_nominal_mean()

            self.mean_nominal = action_samples_init

            # Not aware of boundaries + optional saturation
            # =======================
            # Pre-sample the elements of the random walk:
            self.u_samples[:] = utils.truncated_normal_(self.u_samples) # Truncate beween -2 and +2
            # torch.randn((self.num_trajectories,self.planning_horizon+1,self.dim_action),out=self.u_samples)
            self.u_samples[:] = std_action_samples*self.u_samples

            # Generate samples as (the transpose is important! See https://math.stackexchange.com/questions/2079137/generating-multivariate-normal-samples-why-cholesky)
            self.u_samples[:] = self.u_samples.matmul(self.corr_matrix_chol)

            # Original PDDM method
            # ====================
            # Get the deltas:
            self.n_samples[:] = 0.0
            for t in range(self.planning_horizon):
                self.n_samples[:,t+1,:] = self.beta*self.u_samples[:,t+1,:] + (1.-self.beta)*self.n_samples[:,t,:]

            # Add the deltas to the nominal action sequence:
            # self.action_samples_ii = self.mean_nominal + self.n_samples[:,1:self.planning_horizon+1,:] # We only add the n_samples after the index 0, since those are all zero (initialization as in the paper)
            self.action_samples_ii = self.mean_nominal + self.n_samples[:,0:self.planning_horizon,:] # We only add the n_samples after the index 0, since those are all zero (initialization as in the paper)
            # ====================

            # If some of the actions are out of boundaries, saturate them:
            if self.do_saturate:
                ind_upper_saturate = self.action_samples_ii > self.upper_bound_all
                ind_lower_saturate = self.action_samples_ii < self.lower_bound_all

                # Sanity check:
                if ind_upper_saturate.all():
                    print("@Optimizer PDDM: All actions are crossing the upper bound...")
                    import pdb; pdb.set_trace()
                else:
                    saturations_upper = ind_upper_saturate.sum().item()
                    # print("@Optimizer PDDM: Exceeded upper bound in {0:d} / {1:d} cases ({2:2.2f} %) (!)".format(saturations_upper,self.dim_total,saturations_upper/self.dim_total*100.))

                if ind_lower_saturate.all():
                    print("All actions are crossing the lower bound...")
                    import pdb; pdb.set_trace()
                else:
                    saturations_low = ind_lower_saturate.sum().item()
                    # print("@Optimizer PDDM: Exceeded lower bound in {0:d} / {1:d} cases ({2:2.2f} %) (!)".format(saturations_low,self.dim_total,saturations_low/self.dim_total*100.))

                self.action_samples_ii[ind_upper_saturate] = self.upper_bound_all[ind_upper_saturate]
                self.action_samples_ii[ind_lower_saturate] = self.lower_bound_all[ind_lower_saturate]

                self.saturations_count[0] = saturations_low
                self.saturations_count[1] = saturations_upper

            self.rewards = function(self.action_samples_ii)

            ii += 1

        # Get elite rewards:
        best_optimized_rewards, ind_best_optimized_rewards = torch.topk(self.rewards, k=self.num_elites, dim=0, largest=not minimize) # TODO alonrot: get largest element from values | ~0.06ms for 100 trajectories, and 10 elites
        elites_actions = self.action_samples_ii.index_select(dim=0, index=ind_best_optimized_rewards)

        mean_actions_optimized_elites = torch.mean(elites_actions, dim=0)
        actions_optimized_elites_var = torch.var(elites_actions, unbiased=False, dim=0)

        # import pdb; pdb.set_trace()


        # We update once more the nominal mean:
        if self.use_final_nominal_mean:
            self.update_nominal_mean()


        # print("self.use_final_nominal_mean:",self.use_final_nominal_mean)
        # print("self.max_iters:",self.max_iters)
        # print("self.planning_horizon:",self.planning_horizon)





        # RAD2DEG = 180./np.pi
        # action_samples_np = self.action_samples_ii.cpu().numpy()
        # if self.use_final_nominal_mean:
        #     actions_optimized_elites_mean_np = self.mean_nominal.cpu().numpy()
        # else:
        #     actions_optimized_elites_mean_np = mean_actions_optimized_elites.cpu().numpy()

        # # import pdb; pdb.set_trace()
        # from matplotlib import pyplot as plt
        # ind1 = 0
        # ind2 = 12
        # hdl_fig, hdl_plot = plt.subplots(1,1,figsize=(10,10))
        # for jjj in range(50):
        #     if jjj == 0:
        #         hdl_plot.plot(action_samples_np[jjj,:,ind1]*RAD2DEG,action_samples_np[jjj,:,ind2]*RAD2DEG,color="grey",label="Some trajectories",marker=".",linewidth=1.0)
        #     else:
        #         hdl_plot.plot(action_samples_np[jjj,:,ind1]*RAD2DEG,action_samples_np[jjj,:,ind2]*RAD2DEG,color="grey",marker=".",linewidth=1.0)
        # hdl_plot.plot(actions_optimized_elites_mean_np[:,ind1]*RAD2DEG,actions_optimized_elites_mean_np[:,ind2]*RAD2DEG,color="red",label="Next mean",marker=".",linewidth=1.0)
        # hdl_plot.plot(actions_optimized_elites_mean_np[0,ind1]*RAD2DEG,actions_optimized_elites_mean_np[0,ind2]*RAD2DEG,color="blue",label="Initial point of the selected mean",marker="*")
        
        # if not self.use_final_nominal_mean:
        #     for kkk in range(self.num_elites):
        #         if kkk == 0:
        #             hdl_plot.plot(elites_actions[kkk,:,ind1].cpu().numpy()*RAD2DEG,elites_actions[kkk,:,ind2].cpu().numpy()*RAD2DEG,color="mediumpurple",label="Elites",marker=".",linewidth=1.0)
        #         else:
        #             hdl_plot.plot(elites_actions[kkk,:,ind1].cpu().numpy()*RAD2DEG,elites_actions[kkk,:,ind2].cpu().numpy()*RAD2DEG,color="mediumpurple",marker=".",linewidth=1.0)
            
        # hdl_plot.set_xlim([-30,30])
        # hdl_plot.set_ylim([-30,30])
        # hdl_plot.legend()

        # # plt.show(block=True)
        # if ii > 0:
        #     plt.pause(5.)
        # else:
        #     plt.pause(2.)




        if self.do_saturate:
            self.error_checking_action_solutions_within_bounds(mean_actions_optimized_elites)

        print("@Optimizer PDDM: finished! ii = {0:d} / {1:d}".format(ii,self.max_iters))

        return dict(mean_actions_optimized_elites=mean_actions_optimized_elites, 
                    actions_optimized_elites_var=actions_optimized_elites_var, 
                    best_values_index=ind_best_optimized_rewards, 
                    best_optimized_rewards=best_optimized_rewards, 
                    all_optimized_rewards=self.rewards,
                    saturations_count=self.saturations_count,
                    mean_nominal=self.mean_nominal)

    def error_checking_action_solutions_within_bounds(self,mean_actions_optimized_elites):
        
        # TODO alonrot: added
        bound_low_check = mean_actions_optimized_elites <= self.lower_bound
        if torch.any(bound_low_check):
            print("@Optimizer PDDM: Exceeded lower bound in {0:d} dimensions (!)".format(torch.sum(bound_low_check).item()))
            # import pdb; pdb.set_trace()
        
        bound_high_check = mean_actions_optimized_elites >= self.upper_bound
        if torch.any(mean_actions_optimized_elites >= self.upper_bound):
            print("@Optimizer PDDM: Exceeded higher bound in {0:d} dimensions (!)".format(torch.sum(bound_high_check).item()))
            # import pdb; pdb.set_trace()


class PDDMOptimizer(Optimizer):
    """ Cross-Entropy Optimizer for planning and control.

    See `Deep Reinforcement Learning in a Handful of Trials using Probabilistic
    Dynamics Models <https://arxiv.org/abs/1805.12114>`
    """

    # TODO alonrot: This is a wrapper that involves a call to the true optimizer,
    # CEM, which is the above class

    def __init__(self,
                 device,
                 action_space,
                 planning_horizon,
                 num_trajectories,
                 use_final_nominal_mean,
                 use_correlation_matrix,
                 max_iters,
                 num_elites):
        # compute_return_function is assigned during setup
        self.compute_return_function = None
        self.device = device
        self.num_trajectories = num_trajectories
        # import pdb; pdb.set_trace()
        self.planning_horizon = planning_horizon
        self.dim_action = action_space.shape[0] # TODO alonrot: action_space is Gym dependent! Replace!
        self.last_mean = None
        self.initial_variance = None
        # self.single_lower_bound = torch.from_numpy(action_space.low).to(device=device)  # TODO alonrot: action_space is Gym dependent! Replace!
        # self.single_upper_bound = torch.from_numpy(action_space.high).to(device=device) # TODO alonrot: action_space is Gym dependent! Replace!
        # self.lower_bound = self.single_lower_bound.repeat(self.planning_horizon)
        # self.upper_bound = self.single_upper_bound.repeat(self.planning_horizon)

        self.lower_bound = torch.from_numpy(action_space.low).to(device=device)
        self.upper_bound = torch.from_numpy(action_space.high).to(device=device)

        self.use_final_nominal_mean = use_final_nominal_mean

        # hard coded for now
        self.num_actions = 1
        assert self.num_actions <= self.planning_horizon
        # self.pad_actions = torch.zeros((self.num_actions, self.single_lower_bound.size(0))).to(device=device)
        self.pad_actions = torch.zeros((self.num_actions, self.lower_bound.size(0))).to(device=device)
        self.pddm = PDDM(self.planning_horizon,
                        self.dim_action,
                       self.lower_bound,
                       self.upper_bound,
                       num_trajectories,
                       use_final_nominal_mean,
                       use_correlation_matrix,
                       max_iters,
                       num_elites)

        # import pdb; pdb.set_trace()

        # TODO alonrot added: Pre-allocate memory to keep track of the last set of trajectories
        self.trajectories_last_first_time = True
        self.trajectories_last = None

        self.action_sequence2return = ActionSequence()

        self.action_dim = len(action_space)
        self.action_sequence2return.actions             = torch.zeros(size=(self.planning_horizon,self.action_dim),device=self.device)
        self.action_sequence2return.actions_plan_mean   = torch.zeros(size=(self.planning_horizon,self.action_dim),device=self.device)
        self.action_sequence2return.actions_plan_var    = torch.zeros(size=(self.planning_horizon,self.action_dim),device=self.device)
        self.action_sequence2return.all_optimized_rewards  = torch.zeros((self.num_trajectories,))
        self.action_sequence2return.best_optimized_rewards = torch.zeros((num_elites,))

        self.mean_actions_optimized_elites = torch.zeros(size=(self.planning_horizon,self.action_dim),device=self.device)
        # self.mean_actions_optimized_elites = None

        self.reset_trajectory()


    def reset_trajectory(self):

        # Starting point for the optimizer: actions randomly sampled around the middle point of the interval (same as in CEM, i.e., ignoring temporal correlations)
        mid_bound = (self.lower_bound + self.upper_bound) / 2

        # Variance to sample actions from:
        # var = torch.pow(self.upper_bound - self.lower_bound, 2) / 16.
        # var = torch.pow(self.upper_bound - self.lower_bound, 2) / 25.
        # var = torch.pow(self.lower_bound - self.upper_bound, 2) / 32. # PDDM experiments
        # var = torch.pow(self.lower_bound - self.upper_bound, 2) / 128.
        var = torch.pow(self.lower_bound - self.upper_bound, 2) / 64.
        self.std_action_samples = torch.sqrt(var)

        # Use a constant action sequence as input to the PDDM: Random sequnces will be generated out of it
        self.mean_actions_optimized_elites[:] = mid_bound.repeat(self.planning_horizon, 1)

        # Add noise:
        aux_samples = torch.zeros(size=self.mean_actions_optimized_elites.size()).to(device=self.device)
        aux_samples = utils.truncated_normal_(aux_samples)
        aux_samples = self.std_action_samples*aux_samples
        self.mean_actions_optimized_elites += aux_samples


    def plan_action_sequence(self, state0) -> ActionSequence:

        # TODO alonrot: Function that involves a call to compute_return_function()
        def func(actions):

            assert actions.dim() == 3, "actions: [Ntrajectories x planning_horizon x dim_action]"

            out = self.compute_return_function(state0, actions)

            # import pd; pdb.set_trace()

            # TODO alonrot added: Temporary way of keeping track of the state trajectories used within the optimizer
            if self.trajectories_last_first_time:
                # The first time, we allocate memory. In subsequent accesses, we write on top of it:
                # self.trajectories_last = torch.tensor(size=out.all_trajectories.shape)
                self.action_sequence2return.all_trajectories = out.all_trajectories.clone()
                self.action_sequence2return.all_returns = out.all_returns.clone()
                self.trajectories_last_first_time = False
            else:
                # self.trajectories_last[:,:,:,:] = out.all_trajectories
                self.action_sequence2return.all_trajectories[:] = out.all_trajectories
                self.action_sequence2return.all_returns[:] = out.all_returns

            # import pdb; pdb.set_trace()

            return out.all_returns

        # import pdb; pdb.set_trace()

        # TODO alonrot: Call the actual optimizer:
        solution = self.pddm.optimize(func, self.mean_actions_optimized_elites, self.std_action_samples, minimize=False)

        # TODO alonrot: Data reformatting
        self.mean_actions_optimized_elites[:] = solution['mean_actions_optimized_elites'] # [planning_horizon x dim_action]

        # consume left num_actions and pad with as many zero actions.
        if self.mean_actions_optimized_elites.shape[0] == 1: # If the planning horizon is 1, maybe don't pad:
            import pdb; pdb.set_trace()

        # Consume the first action and add to the end of the planning horizon a randoma action:
        self.mean_actions_optimized_elites[0:-1,:] = self.mean_actions_optimized_elites[1::,:].clone() # Move up all the actions (i.e., consume the first actions)
        
        # Consider sampling here smth close to the last sample in the horizon: It won't be "directed, though"...
        # Since the last and the previous to last samples are both the same, we just add to the last one a sample
        self.mean_actions_optimized_elites[-1,:] += self.std_action_samples*torch.randn(size=(self.dim_action,)).to(device=self.device)
        # import pdb; pdb.set_trace()

        # Take the mean and variance of the last states:
        best_values_index = solution['best_values_index']
        # import pdb; pdb.set_trace()
        state_trajectories = self.action_sequence2return.all_trajectories.index_select(dim=0, index=best_values_index)
        aux = state_trajectories.view(-1,self.planning_horizon+1,len(state0))               # [ Ntraj, planning_horizon+1, Nstates ], with Nstates = len(state0)
        self.action_sequence2return.states_plan_mean = torch.mean(aux, dim=0)               # [ planning_horizon+1, Nstates ], with Nstates = len(state0)
        self.action_sequence2return.states_plan_var = torch.var(aux, unbiased=False, dim=0) # [ planning_horizon+1, Nstates ], with Nstates = len(state0)
        
        # import pdb; pdb.set_trace()

        self.error_checking_device()

        if torch.all(self.mean_actions_optimized_elites == 0.0):
            print("@PDDMOptimizer.plan_action_sequence()")
            import pdb; pdb.set_trace()

        print("self.use_final_nominal_mean: {0:s}".format(str(self.use_final_nominal_mean)))
        if self.use_final_nominal_mean:
            self.action_sequence2return.actions[:] = solution['mean_nominal']
        else:
            self.action_sequence2return.actions[:] = solution['mean_actions_optimized_elites']

        # import pdb; pdb.set_trace()

        self.action_sequence2return.actions_plan_mean[:] = solution['mean_actions_optimized_elites'].view(self.planning_horizon, -1)
        self.action_sequence2return.actions_plan_var[:] = solution['actions_optimized_elites_var'].view(self.planning_horizon, -1)
        self.action_sequence2return.all_optimized_rewards[:] = solution['all_optimized_rewards']
        self.action_sequence2return.best_optimized_rewards[:] = solution['best_optimized_rewards']
        self.action_sequence2return.saturations_count = solution['saturations_count']



        copy2return = ActionSequence()
        copy2return.all_trajectories = self.action_sequence2return.all_trajectories.clone()
        copy2return.all_returns = self.action_sequence2return.all_returns.clone()
        copy2return.actions = self.action_sequence2return.actions.clone()
        copy2return.actions_plan_mean = self.action_sequence2return.actions_plan_mean.clone()
        copy2return.actions_plan_var = self.action_sequence2return.actions_plan_var.clone()
        copy2return.states_plan_mean = self.action_sequence2return.states_plan_mean.clone()
        copy2return.states_plan_var = self.action_sequence2return.states_plan_var.clone()
        copy2return.all_optimized_rewards = self.action_sequence2return.all_optimized_rewards.clone()
        copy2return.best_optimized_rewards = self.action_sequence2return.best_optimized_rewards.clone()
        copy2return.saturations_count = self.action_sequence2return.saturations_count
        if self.action_sequence2return.all_returns_with_particles is not None:
            copy2return.all_returns_with_particles = self.action_sequence2return.all_returns_with_particles.clone()



        return copy2return

    def setup(self, compute_return_function):
        self.compute_return_function = compute_return_function

    def __str__(self):
        return F"PDDMOptimizer (horizon={self.planning_horizon}, " \
            F"num_trajectories={self.num_trajectories}, " \
            F"max iter={self.pddm.max_iters})"


    def error_checking_device(self):

        assert self.action_sequence2return.all_trajectories.device == self.action_sequence2return.all_returns.device
        assert self.action_sequence2return.all_trajectories.device == self.action_sequence2return.actions.device
        assert self.action_sequence2return.all_trajectories.device == self.action_sequence2return.actions_plan_mean.device
        assert self.action_sequence2return.all_trajectories.device == self.action_sequence2return.actions_plan_var.device
        assert self.action_sequence2return.all_trajectories.device == self.action_sequence2return.states_plan_mean.device
        assert self.action_sequence2return.all_trajectories.device == self.action_sequence2return.states_plan_var.device




























    # def optimize(self):



            # # Constrained variance (same as in CEM):
            # lb_dist = torch.min(self.mean_nominal - self.lower_bound,dim=0)[0]
            # ub_dist = torch.min(self.upper_bound - self.mean_nominal,dim=0)[0]
            
            # if torch.all(lb_dist < 0.0) and cc_dbg < 6:
            #     cc_dbg += 1
            #     import pdb;pdb.set_trace()
            
            # if torch.all(ub_dist < 0.0) and cc_dbg < 6:
            #     cc_dbg += 1
            #     import pdb;pdb.set_trace()

            # lb_dist[lb_dist < 0.] = 0.
            # ub_dist[ub_dist > 0.] = 0.
            
            # mstd = torch.sqrt(torch.min(torch.pow(lb_dist / 2, 2), torch.pow(ub_dist / 2, 2)))
            # # import pdb;pdb.set_trace()
            # std_action_samples = torch.sqrt(torch.min(mstd, std_action_samples))
            # # import pdb;pdb.set_trace()





            # # Aware of boundaries
            # # ===================
            # # Sample the elements of the random walk:
            # u_samples = std_action_samples*torch.randn((self.num_trajectories,self.planning_horizon+1,self.dim_action)).to(device=self.device)
            # n_samples = torch.zeros((self.num_trajectories,self.planning_horizon+1,self.dim_action)).to(device=self.device)
            # Nmax_sample_inside = 5
            # for t in range(self.planning_horizon):

            #     # Compute next sample as in the paper:
            #     n_samples[:,t+1,:] = self.beta*u_samples[:,t+1,:] + (1.-self.beta)*n_samples[:,t,:]

            #     # Add boundary correction: Resample if out of the boundaries:
            #     jj = 0; Noutside = 100
            #     while jj < Nmax_sample_inside and Noutside > 0:

            #         new_u_sample = std_action_samples*torch.randn((self.num_trajectories,self.dim_action)).to(device=self.device)
            #         new_n_sample = self.beta*new_u_sample + (1.-self.beta)*n_samples[:,t,:]
            #         Noutside_new = torch.sum( (new_n_sample + self.mean_nominal[t,:] > self.upper_bound) | (new_n_sample + self.mean_nominal[t,:] < self.lower_bound) ).item()
            #         if Noutside_new < Noutside:
            #             n_samples[:,t+1,:] = new_n_sample
            #             Noutside = Noutside_new

            #         jj += 1
            # self.action_samples_ii = self.mean_nominal + n_samples[:,1:self.planning_horizon+1,:] # We only add the n_samples after the index 0, since those are all zero (initialization as in the paper)








            # # Aware of boundaries + Boundary constraints
            # # ===================
            # # Sample the elements of the random walk:
            # u_samples = std_action_samples*torch.randn((self.num_trajectories,self.planning_horizon+1,self.dim_action)).to(device=self.device)
            # n_samples = torch.zeros((self.num_trajectories,self.planning_horizon+1,self.dim_action)).to(device=self.device)
            # Nmax_sample_inside = 5
            # for t in range(self.planning_horizon):

            #     # Compute next sample as in the paper:
            #     n_samples[:,t+1,:] = self.beta*u_samples[:,t+1,:] + (1.-self.beta)*n_samples[:,t,:]

            #     # Add boundary correction: Resample if out of the boundaries:
            #     jj = 0; Noutside = 100
            #     while jj < Nmax_sample_inside and (Noutside > 0 or not sample_forward):

            #         new_u_sample = std_action_samples*torch.randn((self.num_trajectories,self.dim_action)).to(device=self.device)
            #         new_n_sample = self.beta*new_u_sample + (1.-self.beta)*n_samples[:,t,:]

            #         sample_forward = torch.all(torch.sum(new_n_sample*n_samples[:,t,:],dim=1) >= 0.)

            #         Noutside_new = torch.sum( (new_n_sample + self.mean_nominal[t,:] > self.upper_bound) | (new_n_sample + self.mean_nominal[t,:] < self.lower_bound) ).item()
            #         if Noutside_new < Noutside:
            #             n_samples[:,t+1,:] = new_n_sample
            #             Noutside = Noutside_new

            #         jj += 1

            # self.action_samples_ii = self.mean_nominal + n_samples[:,1:self.planning_horizon+1,:] # We only add the n_samples after the index 0, since those are all zero (initialization as in the paper)




            # # Sample forward:
            # # ===================
            # # Sample the elements of the random walk:
            # u_samples = std_action_samples*torch.randn((self.num_trajectories,self.planning_horizon+1,self.dim_action)).to(device=self.device)
            # n_samples = torch.zeros((self.num_trajectories,self.planning_horizon+1,self.dim_action)).to(device=self.device)
            # for t in range(self.planning_horizon):

            #     # Compute next sample as in the paper:
            #     n_samples[:,t+1,:] = self.beta*u_samples[:,t+1,:] + (1.-self.beta)*n_samples[:,t,:]

            #     # Add boundary correction: Resample if out of the boundaries:
            #     jj = 0
            #     Nmax_sample_inside = 10
            #     sample_forward = False
            #     while jj < Nmax_sample_inside and not sample_forward:

            #         new_u_sample = std_action_samples*torch.randn((self.num_trajectories,self.dim_action)).to(device=self.device)
            #         new_n_sample = self.beta*new_u_sample + (1.-self.beta)*n_samples[:,t,:]

            #         sample_forward = torch.all(torch.sum(new_n_sample*n_samples[:,t,:],dim=1) >= 0.)

            #         jj += 1

            # self.action_samples_ii = self.mean_nominal + n_samples[:,1:self.planning_horizon+1,:] # We only add the n_samples after the index 0, since those are all zero (initialization as in the paper)




    # The entore function as of 17th Dec 2019



    # def optimize(self, function, action_samples_init, std_action_samples, minimize=True):

    #     # Error checking:
    #     assert torch.is_tensor(action_samples_init)
    #     assert torch.is_tensor(std_action_samples)
    #     assert self.lower_bound.shape[0] == self.dim_action
    #     assert self.upper_bound.shape[0] == self.dim_action
        
    #     # Reinitialize:
    #     self.mean_nominal[:] = 0.0
    #     self.rewards[:] = 0.0
    #     self.action_samples_ii[:] = 0.0

    #     ii = 0
    #     cc_dbg = 0
    #     while ii < self.max_iters:

    #         # Nominal mean:
    #         if ii == 0:
    #             self.mean_nominal = action_samples_init
    #         else:
    #             self.update_nominal_mean()

    #         # Not aware of boundaries + optional saturation
    #         # =======================
    #         # Pre-sample the elements of the random walk:
    #         torch.randn((self.num_trajectories,self.planning_horizon+1,self.dim_action),out=self.u_samples)
    #         self.u_samples = std_action_samples*self.u_samples


    #         # # Sample forward (used in the last successfully-walking experiment). Useless
    #         # # ===================
    #         # # Sample the elements of the random walk:
    #         # self.n_samples[:] = 0.0
    #         # Nmax_sample_inside = 6
    #         # for t in range(self.planning_horizon):

    #         #     # Add boundary correction: Resample if out of the boundaries:
    #         #     jj = 0
    #         #     sample_forward = False
    #         #     while jj < Nmax_sample_inside and not sample_forward:

    #         #         new_u_sample = std_action_samples*torch.randn((self.num_trajectories,self.dim_action)).to(device=self.device)
    #         #         # self.n_samples[:,t+1,:] = self.beta*new_u_sample + (1.-self.beta)*self.n_samples[:,t,:]
    #         #         self.n_samples[:,t+1,:] = new_u_sample

    #         #         sample_forward = torch.all(torch.sum(self.n_samples[:,t+1,:]*self.n_samples[:,t,:],dim=1) >= 0.)

    #         #         jj += 1


    #         # Brownian motion, sampling forward. Efficient implementation:
    #         # ============================================================
    #         # 
    #         # The implementation below assumes that nor trajectories, nor actions are correlated between them.
    #         # Normally trajectories are assumed uncorrelated.
    #         # A similar implementation style could work if the actions are correlated among themselves.
    #         self.n_samples[:] = 0.0
    #         self.action_samples_ii[:,0,:] = self.mean_nominal[0,:].repeat((self.num_trajectories,1)) # Start with the first action of the nominal mean, repeated across all trajectories
    #         for t in range(self.planning_horizon-1):

    #             # Add boundary correction: Resample if out of the boundaries:
    #             self.ind_rejected[:] = 1 # Set all to True
    #             while self.ind_rejected.sum() > 0:

    #                 # import pdb; pdb.set_trace()
    #                 # print(self.ind_rejected.sum()) # Sanity check
    #                 # Sample a subset with ind_rejected.sum() trajectories, i.e., those that still haven't been accepted
    #                 sample_subset = std_action_samples*torch.randn((self.ind_rejected.sum(),self.dim_action)).to(device=self.device)
    #                 self.n_samples[self.ind_rejected.to(dtype=torch.bool),t+1,:] = sample_subset

    #                 # import pdb; pdb.set_trace()

    #                 # Among the rejected ones, see how many have been accepted, and set them to False.
    #                 # self.ind_rejected is a vector that is meant to shrink at each iteration. Hence, the efficient implementation.
    #                 self.ind_rejected[self.ind_rejected.to(dtype=torch.bool)] = ((self.n_samples[self.ind_rejected.to(dtype=torch.bool),t+1,:]*self.n_samples[self.ind_rejected.to(dtype=torch.bool),t,:]).sum(dim=1) < 0.).to(dtype=torch.int64)

    #             # pdb.set_trace()
    #             assert torch.all((self.n_samples[self.ind_rejected,t+1,:]*self.n_samples[self.ind_rejected,t,:]).sum(dim=1) >= 0.)

    #             self.action_samples_ii[:,t+1,:] = self.action_samples_ii[:,t,:] + self.n_samples[:,t+1,:]

    #         # import pdb; pdb.set_trace()

    #         # self.action_samples_np = self.action_samples_ii.cpu().numpy()
    #         # from matplotlib import pyplot as plt
    #         # hdl_fig, hdl_plot = plt.subplots(1,1)
    #         # for jjj in range(3):
    #         #     hdl_plot.plot(self.action_samples_np[jjj,:,1],self.action_samples_np[jjj,:,3])

    #         # plt.show(block=True)

    #         # ====================


    #         # Original PDDM method
    #         # ====================
    #         # Get the deltas:
    #         # self.n_samples[:] = 0.0
    #         # for t in range(self.planning_horizon):
    #         #     self.n_samples[:,t+1,:] = self.beta*self.u_samples[:,t+1,:] + (1.-self.beta)*self.n_samples[:,t,:]

    #         # # Add the deltas to the nominal action sequence:
    #         # self.action_samples_ii = self.mean_nominal + self.n_samples[:,1:self.planning_horizon+1,:] # We only add the n_samples after the index 0, since those are all zero (initialization as in the paper)
    #         # ====================

    #         # If some of the actions are out of boundaries, saturate them:
    #         if self.do_saturate:
    #             ind_upper_saturate = self.action_samples_ii > self.upper_bound_all
    #             ind_lower_saturate = self.action_samples_ii < self.lower_bound_all

    #             # Sanity check:
    #             if ind_upper_saturate.all():
    #                 print("@Optimizer PDDM: All actions are crossing the upper bound...")
    #                 import pdb; pdb.set_trace()
    #             else:
    #                 saturations_upper = ind_upper_saturate.sum().item()
    #                 # print("@Optimizer PDDM: Exceeded upper bound in {0:d} / {1:d} cases ({2:2.2f} %) (!)".format(saturations_upper,self.dim_total,saturations_upper/self.dim_total*100.))

    #             if ind_lower_saturate.all():
    #                 print("All actions are crossing the lower bound...")
    #                 import pdb; pdb.set_trace()
    #             else:
    #                 saturations_low = ind_lower_saturate.sum().item()
    #                 # print("@Optimizer PDDM: Exceeded lower bound in {0:d} / {1:d} cases ({2:2.2f} %) (!)".format(saturations_low,self.dim_total,saturations_low/self.dim_total*100.))

    #             self.action_samples_ii[ind_upper_saturate] = self.upper_bound_all[ind_upper_saturate]
    #             self.action_samples_ii[ind_lower_saturate] = self.lower_bound_all[ind_lower_saturate]

    #             self.saturations_count[0] = saturations_low
    #             self.saturations_count[1] = saturations_upper

    #         self.rewards = function(self.action_samples_ii)

    #         ii += 1

    #     # Get elite rewards:
    #     best_optimized_rewards, ind_best_optimized_rewards = torch.topk(self.rewards, k=self.num_elites, dim=0, largest=not minimize) # TODO alonrot: get largest element from values | ~0.06ms for 100 trajectories, and 10 elites
    #     elites_rewards = self.action_samples_ii.index_select(dim=0, index=ind_best_optimized_rewards)

    #     mean_actions_optimized_elites = torch.mean(elites_rewards, dim=0)
    #     actions_optimized_elites_var = torch.var(elites_rewards, unbiased=False, dim=0)

    #     # import pdb; pdb.set_trace()


    #     # We update once more the nominal mean:
    #     if self.use_final_nominal_mean:
    #         self.update_nominal_mean()

    #     if self.do_saturate:
    #         self.error_checking_action_solutions_within_bounds(mean_actions_optimized_elites)

    #     print("@Optimizer PDDM: finished! ii = {0:d} / {1:d}".format(ii,self.max_iters))

    #     return dict(mean_actions_optimized_elites=mean_actions_optimized_elites, 
    #                 actions_optimized_elites_var=actions_optimized_elites_var, 
    #                 best_values_index=ind_best_optimized_rewards, 
    #                 best_optimized_rewards=best_optimized_rewards, 
    #                 all_optimized_rewards=self.rewards,
    #                 saturations_count=self.saturations_count,
    #                 mean_nominal=self.mean_nominal)