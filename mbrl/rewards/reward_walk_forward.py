import numpy as np
import numpy.linalg as la
import torch
import sys
import pdb

# alonrot: Solve this problem using either hydra, or installing daisy_toolkit
from tools.tools import get_mbrl_base_path, get_device
mbrl_path = get_mbrl_base_path()
sys.path.append("{0:s}/../daisy_toolkit".format(mbrl_path)) # We look for daisy_toolkit in the same folder where the mbrl repo is placed
from daisy_hardware.daisy_parameters import DaisyConfPETS
daisy_conf = DaisyConfPETS()
daisy_conf.init_transformed_state()
from .reward_base import Reward

class RewardWalkForward(Reward):
    """

    Difference between torch.from_numpy(np_array) and torch.Tensor(np_array):
    In both cases: PyTorch tensors share the memory buffer of NumPy ndarrays. Thus, changing one will be reflected in the other.
    The difference: from_numpy() automatically inherits input array dtype. On the other hand, torch.Tensor is an alias for torch.FloatTensor.
    Source: https://stackoverflow.com/questions/48482787/pytorch-memory-model-torch-from-numpy-vs-torch-tensor
    """


    def __init__(self,dim_state):
        """
    
        """

        assert isinstance(dim_state,int)
        assert dim_state > 0
        self.dim_state = dim_state
        
        # Goal: We want it to walk straight:
        self.goal_base_X = 3.0 # [m]
        self.goal_base_Z_upper_bound = -0.01 # [m] # We only penalize being below this value, while if above, we don't add reward
        
        self.goal_sin_phiX = 0.0 # []
        self.goal_sin_phiY = 0.0 # []

        self.weight_base_X = 100./3.
        self.weight_base_Z = 1./0.1

        self.weigth_sin_phiX = 10/0.35
        self.weigth_sin_phiY = 10/0.35

        self.indices_trans_of = daisy_conf.indices_trans_of

        # This flag needs to be set when entering the reward function.
        # It will be inferred from the state variable
        self.work_with_tensor = None

    def get_reward_signal(self,state_curr,action_curr=None):

        # Check whether we got a tensor or a numpy array
        self.work_with_tensor = False
        if torch.is_tensor(state_curr):
            self.work_with_tensor = True

        self.error_checking_input(state_curr,action_curr)

        # Define current state:
        pos_base_X = state_curr[:,self.indices_trans_of["base_position"][0]] # X
        pos_base_Z = state_curr[:,self.indices_trans_of["base_position"][2]] # Z

        ori_base_sin_phiX = state_curr[:,self.indices_trans_of["base_orientation"][1]] # sin(phiX)
        ori_base_sin_phiY = state_curr[:,self.indices_trans_of["base_orientation"][3]] # sin(phiY)

        cost_distance_vec = self.weight_base_X*abs(self.goal_base_X - pos_base_X) + \
                            self.weigth_sin_phiX*abs(ori_base_sin_phiX) + \
                            self.weigth_sin_phiY*abs(ori_base_sin_phiY)

        if self.work_with_tensor:
            cost_distance_vec += self.weight_base_Z*(self.goal_base_Z_upper_bound - pos_base_Z).clamp(min=0) # Equivalent to max(Zgoal - Zcurr,0)
        else:
            cost_distance_vec += self.weight_base_Z*(self.goal_base_Z_upper_bound - pos_base_Z).clip(min=0)


        reward = -cost_distance_vec

        self.error_checking_reward(reward,state_curr.shape[0])

        return reward

    def error_checking_input(self,state_curr,action_curr=None):

        # Error checking:
        if self.work_with_tensor:
            assert state_curr.dim() == 2
        else:
            assert state_curr.ndim == 2
        assert state_curr.shape[0] > 0
        assert state_curr.shape[1] == self.dim_state

    def error_checking_reward(self,reward,Ntrajectories):

        if self.work_with_tensor:
            assert reward.dim() == 1
        else:
            assert reward.ndim == 1
        assert reward.shape[0] == Ntrajectories

    def add_normalization(self):
        pass



