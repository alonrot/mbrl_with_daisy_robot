import logging
import numpy as np
import torch

from mbrl import utils
from .optimizer import Optimizer, ActionSequence

# from mbrl.policies.sine_waves_policy import ActionSequenceSines
from mbrl.policies.sine_waves_policy import SineWavesPolicyParametrized

from datetime import datetime

log = logging.getLogger(__name__)

class CEM:
    """ Cross-Entropy Method (CEM) for optimization

    See `The Cross-Entropy Method for Optimization
    <https://people.smp.uq.edu.au/DirkKroese/ps/CEopt.pdf>`
    """
    def __init__(self,
                 solution_dim,
                 lower_bound,
                 upper_bound,
                 num_trajectories,
                 num_elites,
                 max_iters,
                 mean_alpha,
                 var_alpha,
                 epsilon):
        self.solution_dim = solution_dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.num_trajectories = num_trajectories
        self.num_elites = num_elites
        self.max_iters = max_iters
        self.mean_alpha = mean_alpha
        self.var_alpha = var_alpha
        self.epsilon = epsilon
        self.device = self.lower_bound.device
        self.debug_level = 0

    def optimize(self, function, initial_mean, initial_variance, minimize=True):
        assert torch.is_tensor(initial_mean)
        assert torch.is_tensor(initial_variance)

        mean = initial_mean
        var = initial_variance
        n = 0
        means = []
        variances = []
        best_values_avgs = []
        samples = torch.zeros(self.num_trajectories, self.solution_dim).to(device=self.device)

        while n < self.max_iters and var.max().item() > self.epsilon:
            lb_dist = mean - self.lower_bound
            ub_dist = self.upper_bound - mean
            assert torch.all(lb_dist > 0.0), "Distance to bound should be positive!!"
            assert torch.all(ub_dist > 0.0), "Distance to bound should be positive!!"
            mv = torch.min(torch.pow(lb_dist / 2, 2), torch.pow(ub_dist / 2, 2))
            constrained_var = torch.min(mv, var)

            # Samples is originally zeros; this function returns a sample from a truncated normal
            # It assumes that that samples are uncorrelated, though, and mean=0, std=1
            samples = utils.truncated_normal_(samples) # Truncate a N(0,1) by cutting from -2 to 2
            samples = samples * torch.sqrt(constrained_var) + mean

            # time_init_function_call = datetime.utcnow().timestamp()
            values = function(samples) # ~16ms for 100 trajectories, and planning horizon of 5 steps
            # time_end_function_call = datetime.utcnow().timestamp()
            # print("Trajectories roll-out time: {0:2.2f} [ms]".format((time_end_function_call-time_init_function_call)*1000))

            best_values, best_values_index = torch.topk(values, k=self.num_elites, dim=0, largest=not minimize) # TODO alonrot: get largest element from values | ~0.06ms for 100 trajectories, and 10 elites
            elites = samples.index_select(dim=0, index=best_values_index)
            new_mean = torch.mean(elites, dim=0)
            new_var = torch.var(elites, unbiased=False, dim=0)

            # # Original implementation: requires self.mean_alpha -> 1.0 for small steps. Disagrees with literature
            # mean = self.mean_alpha * mean + (1 - self.mean_alpha) * new_mean
            # var = self.var_alpha * var + (1 - self.var_alpha) * new_var

            # alonrot: New implementation: requires self.mean_alpha -> 0.0 for small steps. AGREEES with literature
            mean = self.mean_alpha * new_mean + (1 - self.mean_alpha) * mean
            var = self.var_alpha * new_var + (1 - self.var_alpha) * var

            means.append(mean)
            variances.append(var)
            best_values_avgs.append(best_values.mean())

            if self.debug_level >= 2:
                # evaluating function can be expensive, only do it if we really want to debug
                log.info(f"Iter {n}, value={function(new_mean)}, max var={var.max().item()}")
            n += 1

        if self.debug_level >= 1:
            # evaluating function can be expensive, only do it if we really want to debug
            log.info(f"Final result value={function(mean).item()}, max var={var.max().item()}")

        print("Optimizer CEM finished!    n = "+str(n)+ " / "+str(self.max_iters) + " ;;; var.max().item() = " + str(var.max().item()) + " , required = " + str(self.epsilon))
        print("Optimizer CEM finished!    self.num_elites = " + str(self.num_elites))

        # alonrot added:
        if not (mean <= self.lower_bound).sum() == 0:
            import pdb; pdb.set_trace()
        if not (mean >= self.upper_bound).sum() == 0:
            import pdb; pdb.set_trace()

        diag = dict(var=var, means=means, variances=variances, best_values_avgs=best_values_avgs, best_values_index=best_values_index, best_optimized_rewards=best_values, all_optimized_rewards=values)
        return dict(result=mean, var=var, diag=diag)


class CEMOptimizer(Optimizer):
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
                 num_elites,
                 max_iters,
                 mean_alpha,
                 var_alpha,
                 epsilon):
        
        # compute_return_function is assigned during setup
        self.compute_return_function = None
        self.device = device
        self.num_trajectories = num_trajectories
        self.planning_horizon = planning_horizon
        self.solution_dim = planning_horizon * action_space.shape[0] # TODO alonrot: action_space is Gym dependent! Replace!
        self.last_mean = None
        self.initial_variance = None
        self.single_lower_bound = torch.from_numpy(action_space.low).to(device=device)  # TODO alonrot: action_space is Gym dependent! Replace!
        self.single_upper_bound = torch.from_numpy(action_space.high).to(device=device) # TODO alonrot: action_space is Gym dependent! Replace!
        self.lower_bound = self.single_lower_bound.repeat(self.planning_horizon)
        self.upper_bound = self.single_upper_bound.repeat(self.planning_horizon)
        self.num_actions2pad = 1 # hard coded for now
        assert self.num_actions2pad <= self.planning_horizon
        self.pad_actions = torch.zeros((self.num_actions2pad, self.single_lower_bound.size(0))).to(device=device)
        self.cem = CEM(self.solution_dim,
                       self.lower_bound,
                       self.upper_bound,
                       num_trajectories,
                       num_elites,
                       max_iters,
                       mean_alpha,
                       var_alpha,
                       epsilon)
        self.reset_trajectory()

        # TODO alonrot added: Pre-allocate memory to keep track of the last set of trajectories
        self.trajectories_last_first_time = True
        self.trajectories_last = None
        # self.trajectories_last = torch.tensor(( self.num_trajectories,
        #                                         self.particles,
        #                                         self.planning_horizon + 1, # Needs to be passed in the constructor
        #                                         self.state_dim))          # Needs to be passed in the constructor
        self.action_sequence2return = ActionSequence()

        self.action_dim = len(action_space)
        self.action_sequence2return.actions             = torch.zeros(size=(self.planning_horizon,self.action_dim),device=self.device)
        self.action_sequence2return.actions_plan_mean   = torch.zeros(size=(self.planning_horizon,self.action_dim),device=self.device)
        self.action_sequence2return.actions_plan_var    = torch.zeros(size=(self.planning_horizon,self.action_dim),device=self.device)
        self.action_sequence2return.all_optimized_rewards  = torch.zeros((self.num_trajectories,))
        self.action_sequence2return.best_optimized_rewards = torch.zeros((num_elites,))

    def reset_trajectory(self):
        mid_bound = (self.lower_bound + self.upper_bound) / 2
        self.last_mean = mid_bound

        # TODO: try to eliminate 16 here by using a better var_alpha
        var = torch.pow(self.lower_bound - self.upper_bound, 2) / 16
        # var = torch.pow(self.lower_bound - self.upper_bound, 2) / 64
        self.initial_variance = var

    def plan_action_sequence(self, state0) -> ActionSequence:

        # TODO alonrot: Function that involves a call to compute_return_function()
        def func(a):
            if a.dim() == 2:
                # TODO alonrot: .view() Returns a new tensor with the same data as the self tensor but of a different shape
                actions = a.view(self.num_trajectories, self.planning_horizon, -1) # [Ntraj x Time horizon x Nactions]
            elif a.dim() == 1:
                actions = a.view(1, self.planning_horizon, -1)
            else:
                raise ValueError("Unexpected size for a")

            # alonrot: self.compute_return_function() points to:
            #          pets.petspolicy.PETSPolicy.compute_return()
            #          and it's defined inside pets.petspolicy.PETSPolicy.setup() by calling self.optimizer.setup(self.compute_return)
            out = self.compute_return_function(state0, actions)

            # TODO alonrot added: Temporary way of keeping track of the state trajectories used within the optimizer
            if self.trajectories_last_first_time:
                # The first time we allocate memory. In subsequent accesses, we write on top of it:
                # self.trajectories_last = torch.tensor(size=out.all_trajectories.shape)
                self.action_sequence2return.all_trajectories = out.all_trajectories.clone()
                self.action_sequence2return.all_returns = out.all_returns.clone()
                self.trajectories_last_first_time = False
            else:
                self.action_sequence2return.all_trajectories[:] = out.all_trajectories
                self.action_sequence2return.all_returns[:] = out.all_returns


            return out.all_returns


        # Call the actual optimizer:
        solution = self.cem.optimize(func, self.last_mean, self.initial_variance, minimize=False)

        # Data reformatting
        result = solution['result'].view(self.planning_horizon, -1) # result is just the mean
        # asserts actions are within bounds
        assert (result <= self.single_lower_bound).sum() == 0
        assert (result >= self.single_upper_bound).sum() == 0

        # Take the mean and variance of the last states:
        best_values_index = solution['diag']['best_values_index']
        state_trajectories = self.action_sequence2return.all_trajectories.index_select(dim=0, index=best_values_index)
        aux = state_trajectories.view(-1,self.planning_horizon+1,len(state0))               # [ Ntraj, planning_horizon+1, Nstates ], with Nstates = len(state0)
        self.action_sequence2return.states_plan_mean = torch.mean(aux, dim=0)               # [ planning_horizon+1, Nstates ], with Nstates = len(state0)
        self.action_sequence2return.states_plan_var = torch.var(aux, unbiased=False, dim=0) # [ planning_horizon+1, Nstates ], with Nstates = len(state0)
        

        assert self.action_sequence2return.all_trajectories.device == self.action_sequence2return.all_returns.device
        assert self.action_sequence2return.all_trajectories.device == self.action_sequence2return.actions.device
        assert self.action_sequence2return.all_trajectories.device == self.action_sequence2return.actions_plan_mean.device
        assert self.action_sequence2return.all_trajectories.device == self.action_sequence2return.actions_plan_var.device
        assert self.action_sequence2return.all_trajectories.device == self.action_sequence2return.states_plan_mean.device
        assert self.action_sequence2return.all_trajectories.device == self.action_sequence2return.states_plan_var.device

        # consume left num_actions and pad with as many zero actions.
        if result.shape[0] == 1: # If the planning horizon is 1, maybe don't pad:
            import pdb; pdb.set_trace()

        if result.shape[0] > 1:
            """
            Padding now with the center of the boundaries
            """
            
            result_padded = result.clone()
            aux = result_padded[self.num_actions2pad::,:]
            result_padded[0:-self.num_actions2pad,:] = aux.clone()
            result_padded[-self.num_actions2pad::,:] = 0.5*(self.single_lower_bound + self.single_upper_bound)

            # alonrot added :
            # self.last_mean = torch.cat((result.narrow(0, 1, result.size(0) - self.num_actions2pad), self.pad_actions)) # original padding
            # padding with zero actions is a bad idea, as it can happen that the new mean is OUT of the action space limits. For example,
            # if the limits of one of the actions are [+1, +5], any zero action is out of it.
            # To solve this, we pad with the middle of the interval. Also, the padding above allocates new memory, which can be avoided
            # import pdb; pdb.set_trace()
            # self.last_mean[0:-self.num_actions2pad,:] = self.last_mean[self.num_actions2pad::,:]
            # import pdb; pdb.set_trace()
            # self.last_mean[-self.num_actions2pad::,:] = 0.5*(self.single_lower_bound + self.single_upper_bound)


        # flatten vector, cem does not care
        self.last_mean = result_padded.view(-1)
        # self.last_mean = self.last_mean.view(-1)

        if torch.all(self.last_mean == 0.0):
            print("@CEMOptimizer.plan_action_sequence()")
            import pdb; pdb.set_trace()

        self.action_sequence2return.actions[:] = solution['result'].view(self.planning_horizon, -1)
        self.action_sequence2return.actions_plan_mean[:] = solution['result'].view(self.planning_horizon, -1)
        self.action_sequence2return.actions_plan_var[:] = solution['var'].view(self.planning_horizon, -1)
        self.action_sequence2return.all_optimized_rewards[:] = solution['diag']['all_optimized_rewards']
        self.action_sequence2return.best_optimized_rewards[:] = solution['diag']['best_optimized_rewards']

        return self.action_sequence2return

    def setup(self, compute_return_function):
        self.compute_return_function = compute_return_function

    def __str__(self):
        return F"CEMOptimizer (horizon={self.planning_horizon}, " \
            F"num_trajectories={self.num_trajectories}, " \
            F"max iter={self.cem.max_iters})"


class CEMOptimizerPETS_parametrized(Optimizer):
    """ Cross-Entropy Optimizer for planning and control.

    See `Deep Reinforcement Learning in a Handful of Trials using Probabilistic
    Dynamics Models <https://arxiv.org/abs/1805.12114>`
    """

    # TODO alonrot: This is a wrapper that involves a call to the true optimizer,
    # CEM, which is the above class

    def __init__(self,
                 device,
                 parameter_space,
                 planning_horizon,
                 dt,
                 action_dim,
                 num_trajectories,
                 num_elites,
                 max_iters,
                 mean_alpha,
                 var_alpha,
                 epsilon):
        # compute_return_function is assigned during setup
        self.compute_return_function = None
        self.device = device
        self.num_trajectories = num_trajectories
        self.planning_horizon = planning_horizon
        self.action_dim = action_dim
        # import pdb; pdb.set_trace()
        # self.solution_dim = planning_horizon * parameter_space.shape[0] # TODO alonrot: action_space is Gym dependent! Replace!
        self.solution_dim = parameter_space.shape[0] # TODO alonrot: action_space is Gym dependent! Replace!
        self.last_mean = None
        self.initial_variance = None
        self.single_lower_bound = torch.from_numpy(parameter_space.low).to(device=device)  # TODO alonrot: parameter_space is Gym dependent! Replace!
        self.single_upper_bound = torch.from_numpy(parameter_space.high).to(device=device) # TODO alonrot: parameter_space is Gym dependent! Replace!
        self.lower_bound = self.single_lower_bound
        self.upper_bound = self.single_upper_bound
        # self.lower_bound = self.single_lower_bound.repeat(self.planning_horizon)
        # self.upper_bound = self.single_upper_bound.repeat(self.planning_horizon)
        # hard coded for now
        self.num_actions2pad = 1
        assert self.num_actions2pad <= 1 # TODO alonrot: added, in the parametrized case, the parameters are given as a vector, and thus, self.num_actions2pad can't be larger than 1
        self.pad_actions = torch.zeros((self.num_actions2pad, self.lower_bound.size(0))).to(device=device)
        self.cem = CEM(self.solution_dim,   # -> Since here we're dealing with parameter space directly, self.solution_dim should be equal to parameter space,
                                            # rather than action_space x planning_horizon
                       self.lower_bound,
                       self.upper_bound,
                       num_trajectories,
                       num_elites,
                       max_iters,
                       mean_alpha,
                       var_alpha,
                       epsilon)
        self.reset_trajectory()

        # import pdb; pdb.set_trace()

        # We can pass from hydra SineWaves and everything

        # Get sample from action space: This can be a dummy sample, as we just need to initialize the class:
        pars_sample = np.random.uniform(low=parameter_space.low,high=parameter_space.high)
        self.CPG_policy = SineWavesPolicyParametrized(device, parameter_space, planning_horizon, action_dim)
        self.dt = dt
        # self.action_sequence_sines = ActionSequenceSines(pars_sample[0],pars_sample[1],pars_sample[2],pars_sample[3])

        # The action sequnce generates a sequence of actions, given a chosen set of parameters freq_vec, ampl_vec, etc. for all the joints
        # We need indeed the SineWavesPolicy, as this one takes care of mapping the policy parameters (which can be anything), to common sine 
        # parameters

        # # Time vector:
        # self.t_vec = np.arange(0,self.planning_horizon) * dt

        # Actions:
        self.actions = torch.zeros((self.num_trajectories,self.planning_horizon,self.action_dim))

    def reset_trajectory(self):
        mid_bound = (self.lower_bound + self.upper_bound) / 2
        self.last_mean = mid_bound

        # TODO: try to eliminate 16 here by using a better var_alpha
        var = torch.pow(self.lower_bound - self.upper_bound, 2) / 16
        # var = torch.pow(self.lower_bound - self.upper_bound, 2) / 32
        self.initial_variance = var

    def plan_action_sequence(self, state0, t_curr) -> ActionSequence:

        # TODO alonrot: Function that involves a call to compute_return_function()
        def objective_function(params):

            # # Error checking:
            # assert params.shape[0] == self.solution_dim

            # params: [Ntrajectories x 1 x Nparameters]

            params_reshaped = params.view(self.num_trajectories, 1, -1) # [Ntraj x Time horizon x Nactions]

            for k in range(params_reshaped.shape[0]):

                # Recompute sinusoidals:
                self.CPG_policy.update_policy(params_reshaped[k,0,:])

                # Compute actions:
                actions_per_traj = self.CPG_policy.action_sequence_sines.generate_action_sequence(state0, self.dt, t_curr)
                self.actions[k,:,:] = torch.from_numpy(actions_per_traj).to(device=self.device)

            # Generate a sequence of self.actions using the parameters:

            # TODO alonrot: self.compute_return_function() points to:
            # TODO          pets.petspolicy.PETSPolicy.compute_return()
            # TODO          and it's defined inside pets.petspolicy.PETSPolicy.setup() by calling self.optimizer.setup(self.compute_return)
            # import pdb; pdb.set_trace()
            out = self.compute_return_function(state0, self.actions)
            return out.all_returns

        # TODO alonrot: Call the actual optimizer:
        solution = self.cem.optimize(objective_function, self.last_mean, self.initial_variance, minimize=False)

        # TODO alonrot: Data reformatting
        # result = solution['result'].view(self.planning_horizon, -1)
        result = solution['result'].view(1, -1) # TODO alonrot: Result is now in parameter space
        # asserts actions are within bounds
        # assert (result <= self.single_lower_bound).sum() == 0, "The results exceedes the lower bound"
        # assert (result >= self.single_upper_bound).sum() == 0, "The results exceedes the upper bound"
        if not (result <= self.single_lower_bound).sum() == 0:
            import pdb; pdb.set_trace()
        if not (result >= self.single_upper_bound).sum() == 0:
            import pdb; pdb.set_trace()
        
        # consume left num_actions and pad with as many zero actions.
        # TODO alonrot: What is this about?? -> In my case, it's resetting self.last_mean to all zeroes...
        # self.last_mean = torch.cat((result.narrow(0, 1, result.size(0) - self.num_actions2pad), self.pad_actions))
        
        # flatten vector, cem does not care
        self.last_mean = self.last_mean.view(-1)
        
        # import pdb; pdb.set_trace()

        # alonrot added: Compute the action sequence corresponding to the optimized set of parameters:
        self.CPG_policy.update_policy(result)
        self.CPG_policy.action_sequence_sines.generate_action_sequence(state0, self.dt, t_curr) # This populates the internal state self.actions_plan
        return self.CPG_policy.action_sequence_sines

        # # TODO alonrot: Return as an ActionSequence() object
        # ret = ActionSequence()
        # ret.actions = result
        # return ret

    def setup(self, compute_return_function):
        self.compute_return_function = compute_return_function

    def __str__(self):
        return F"CEMOptimizer (horizon={self.planning_horizon}, " \
            F"num_trajectories={self.num_trajectories}, " \
            F"max iter={self.cem.max_iters})"





