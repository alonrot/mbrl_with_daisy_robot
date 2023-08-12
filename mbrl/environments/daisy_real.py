from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from omegaconf import OmegaConf
import numpy as np
import torch
import pdb
import sys
import os
from pets.conf.env.daisy.optimizer.search_space_limits import CPG_limits4PETS_as_box, joint_limits4PETS_as_box, joint_limits4PETS_as_box_for_walking_tight
# from mbrl.rewards import Reward4NNanalysis
from tools.tools import DEG2RAD
import time
import logging
logging._srcfile = None
logging.logThreads = 0
logging.logProcesses = 0
# logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
# Do not import logging.handlers nor logging.config unless really needed

# alonrot: Solve this problem using either hydra, or installing daisy_toolkit
from tools.tools import get_mbrl_base_path, GymBox
mbrl_path = get_mbrl_base_path()
sys.path.append("{0:s}/../daisy_toolkit".format(mbrl_path)) # We look for daisy_toolkit in the same folder where the mbrl repo is placed
# from daisy_hardware.daisy_hard_base import DaisyRobotDirect
from daisy_hardware.daisy_hard_network import DaisyRobotNetwork
from daisy_hardware.daisy_hard_base import DaisyRobotDirect
from daisy_hardware.daisy_parameters import DaisyConfPETS

class DaisyRealRobotEnv():

    def __init__(self,cfg_env): 

        # Transform into a dictionary, according to the convention established in DaisyConfPETS (daisy_toolkit.daisy_hardware.daisy_parameters.DaisyConfPETS)
        self.daisy_conf = DaisyConfPETS()
        self.daisy_conf.init_error_codes()
        self.state_dict = self.daisy_conf.copy_dict()
        self.DIM_STATE = self.daisy_conf.DIM_STATE # Raw state
        self.DIM_STATE_TRANS = self.daisy_conf.DIM_STATE_TRANS # State that will be used by MBRL (potentially with a different size)
        self.DIM_POS = self.daisy_conf.DIM_STATE
        self.state_raw = np.zeros(self.daisy_conf.DIM_STATE)
        
        # Transformed state, for convenience to the NN training. We'll have both, the normal state and this one
        self.state_transformed = np.zeros(self.DIM_STATE_TRANS)

        # It might be possible to avoid this using hydra:
        if cfg_env.which_interface == "direct":
            self.daisy_interface = DaisyRobotDirect(
                                    freq_fbk=cfg_env.interface.direct.freq_fbk,
                                    use_fake_vision_debug=cfg_env.interface.direct.use_fake_vision_debug,
                                    use_fake_robot_debug=cfg_env.interface.direct.use_fake_robot_debug,
                                    junk_matrix_vision_length=cfg_env.interface.direct.junk_matrix_vision_length,
                                    freq_hold_position=cfg_env.interface.direct.hold_position_process.freq,
                                    name_hold_position=cfg_env.interface.direct.hold_position_process.name_proc,
                                    dim_state=cfg_env.interface.direct.dim_state,
                                    time2pos_reset=cfg_env.interface.direct.time2pos_reset,
                                    freq_action_reset=cfg_env.interface.direct.freq_action_reset)
        
        elif cfg_env.which_interface == "network":
            self.daisy_interface = DaisyRobotNetwork(
                                HOST=cfg_env.interface.network.HOST,
                                PORT_DES_POSITIONS=cfg_env.interface.network.PORT_DES_POSITIONS,
                                PORT_CURR_POSITIONS=cfg_env.interface.network.PORT_CURR_POSITIONS,
                                PORT_ACK=cfg_env.interface.network.PORT_ACK,
                                PORT_FLAG_RESET=cfg_env.interface.network.PORT_FLAG_RESET,
                                PORT_STATUS_ROBOT=cfg_env.interface.network.PORT_STATUS_ROBOT,
                                UDP_IP=cfg_env.interface.network.UDP_IP,
                                buff=cfg_env.interface.network.buff,
                                DIM_STATE=self.DIM_STATE, # Raw state
                                freq_action_reset=cfg_env.interface.network.freq_action_reset,
                                ask_user_input_reset=cfg_env.interface.network.ask_user_input_reset,
                                time2pos_reset=cfg_env.interface.network.time2pos_reset)
        else:
            raise ValueError("cfg_env_comm.which_interface = {interface.direct,interface_network}")

        # Parse inputs and error checking:
        cfg_env_comm = cfg_env.interface.network
        assert isinstance(cfg_env_comm.time2sleep_after_reset,int) or isinstance(cfg_env_comm.time2sleep_after_reset,float)
        assert cfg_env_comm.time2sleep_after_reset >= 0.0 or cfg_env_comm.time2sleep_after_reset == -1
        self.time2sleep_after_reset = cfg_env_comm.time2sleep_after_reset

        assert isinstance(cfg_env_comm.reset_type,str), "Reset type must be {stand_up,legs_extended,legs_extended_directly}"
        assert cfg_env_comm.reset_type == "legs_extended" or cfg_env_comm.reset_type == "legs_extended_directly" or cfg_env_comm.reset_type == "stand_up" or cfg_env_comm.reset_type == "stand_up_directly", "Reset type must be {stand_up,legs_extended,legs_extended_directly,stand_up_directly}"
        self.reset_type = cfg_env_comm.reset_type

        # Have here a reward initialization, where we initialize different kinds of rewards:
        self.goal_xy = np.array([2.,2.])
        self.ind_basepos_xy = np.array([18,19],dtype=int)

        # TODO alonrot: pass to DaisyRobotDirect also a list with the desired observations, i.e., 
        # We want: joint_angular_pos, base_position, base_orientation
        # NOTE: This is very involved, and not so necessary to move forward, so Low priority

        # joint_lims_low, joint_lims_high = joint_limits4PETS_as_box_for_walking()
        joint_lims_low, joint_lims_high = joint_limits4PETS_as_box_for_walking_tight()
        self.action_space = GymBox.create_from_vec(joint_lims_low,joint_lims_high)

        # Limits of the parameter space (only needed if we use a parametrized policy):
        lim_box_low, lim_box_high, ind_box_dict = CPG_limits4PETS_as_box()
        self.parameter_space = GymBox.create_from_vec(lim_box_low,lim_box_high,ind_box_dict)
        # self.parameter_space = GymBox(lim_low,lim_high,self.get_action_dim())

        self.check_still_standing = cfg_env_comm.check_still_standing
        if self.check_still_standing.use:
            assert isinstance(self.check_still_standing.base_position_Z_thres,float)
            self.base_position_Z_thres = self.check_still_standing.base_position_Z_thres

        self.check_still_not_flipped = cfg_env_comm.check_still_not_flipped
        if self.check_still_not_flipped.use:
            self.flip_limit = flip_limit

        self.check_still_not_flipped = cfg_env_comm.check_still_not_flipped

        self.zero_offset_observations = cfg_env_comm.zero_offset_observations
        # self.state_offset = np.zeros(len(self.ind_selector))
        self.state_offset = np.zeros(self.DIM_STATE)
        
        # Stabilization:
        assert isinstance(cfg_env_comm.wait4stabilization,bool)
        self.wait4stabilization = cfg_env_comm.wait4stabilization
        if self.wait4stabilization:
            self.init_wait4measurements_stabilization(cfg_env_comm)

        # Other members:
        self.np_random = None # TODO alonrot: Not sure what this is yet...
        self.reward_func = None
        self.device = cfg_env_comm.device

    def setup_reward_function(self,reward_func):
        self.reward_func = reward_func

    def step(self, action):

        assert isinstance(action,np.ndarray)

        # Return state and observations:
        # state, status_robot = super().step(action) # Call DaisyRobotDirect.step()
        state, status_robot = self.daisy_interface.step(action) # Call DaisyRobotDirect.step()
        print("@daisy_real.step(): status_robot = {0:2.2f}".format(status_robot))

        # Interpret robot status. If there was an error, report is_alive = False, so that the episode can stop in the upper levels:
        is_in_good_status = self.get_robot_is_alive_from_status(status_robot)

        self.collect_observations_and_update(state)
        
        # We compute the reward using the state measurements:
        reward = self.reward_func(self.get_state_transformed()[None,:])
        print("Reward next state env.step(): ",reward)

        # Identify if the robot is in the ground:
        is_standing = True
        if is_in_good_status == True and self.check_still_standing.use:
            is_standing = self.is_robot_still_standing()
        
        # Identify if the robot is too tilted:
        is_not_flipped = True
        if is_in_good_status == True and self.check_still_not_flipped.use:
            is_not_flipped = self.is_robot_not_flipped()

        is_alive = is_standing and is_not_flipped and is_in_good_status
        print("@daisy_real.step(): is_alive = {0:s}".format(str(is_alive)))

        return self.get_state_transformed(), reward, is_alive

    def update_offsets(self,state):
        # assert self.which_observations == "full"
        # self.state_offset[self.daisy_conf.ind_pose] = state[self.daisy_conf.ind_pose] # This assumes that the robo pose estimation has been estabilized
        self.state_offset[:] = 0.0
        self.state_offset[self.daisy_conf.indices_of["base_pose"]] = state[self.daisy_conf.indices_of["base_pose"]]   # This assumes that the robot pose estimation has been estabilized, 
                                                                                                                                    # i.e., that the reset function has been called, with self.wait4stabilization = True
        
        # # Further treatment: The X rotation is near +90 at initialization, so, we reset it to zero: -> NOT NEEDED. Simply define the offset as the current state
        # self.state_offset[self.daisy_conf.indices_of["base_orientation"][0]] += 0.5*np.pi
        
    def collect_observations_and_update(self,state):

        # Update dictionary:
        self.daisy_conf.fill_dict(state-self.state_offset,self.state_dict) # Fill state dictionary with state_raw
        
        # Update state raw:
        self.state_raw[:] = state

        # Update state transformed:
        self.state_transformed[self.daisy_conf.indices_trans_of["joint_angular_pos"]] = self.state_dict["joint_angular_pos"]
        self.state_transformed[self.daisy_conf.indices_trans_of["base_position"]] = self.state_dict["base_position"]
        self.state_transformed[self.daisy_conf.indices_trans_of["base_orientation"][0]] = np.cos(self.state_dict["base_orientation"][0])
        self.state_transformed[self.daisy_conf.indices_trans_of["base_orientation"][1]] = np.sin(self.state_dict["base_orientation"][0])
        self.state_transformed[self.daisy_conf.indices_trans_of["base_orientation"][2]] = np.cos(self.state_dict["base_orientation"][1])
        self.state_transformed[self.daisy_conf.indices_trans_of["base_orientation"][3]] = np.sin(self.state_dict["base_orientation"][1])
        self.state_transformed[self.daisy_conf.indices_trans_of["base_orientation"][4]] = np.cos(self.state_dict["base_orientation"][2])
        self.state_transformed[self.daisy_conf.indices_trans_of["base_orientation"][5]] = np.sin(self.state_dict["base_orientation"][2])
        # pdb.set_trace()
        self.state_transformed[self.daisy_conf.indices_trans_of["joint_angular_vel"]] = self.state_dict["joint_angular_vel"]

    def seed(self,my_seed):
        pass

    def reset(self):
        """
        For compatibility with mbrl/pets (more specifically, with mbrl.utils.sample_episode(), 
        this reset function has to return the initial state of the robot, in a vector)
        """

        # Call to DaisyRobotDirect.reset(), which executes a series of movements in the robot
        # to bring it to the home position:
        position_joint_curr = self.read_angular_position()

        # Call the reset function, and retrieve the status of the robot:
        # state_after_reset, status_robot = super().reset(position_joint_curr,reset_type=self.reset_type,time2sleep_after_reset=self.time2sleep_after_reset)
        state_after_reset, status_robot = self.daisy_interface.reset(position_joint_curr,reset_type=self.reset_type,time2sleep_after_reset=self.time2sleep_after_reset)
        print("@daisy_real.reset(): status_robot = {0:2.2f}".format(status_robot))

        # Interpret robot status. If there was an error, report is_alive = False, so that the episode can stop in the upper levels:
        is_alive = self.get_robot_is_alive_from_status(status_robot)
        if not is_alive:
            return state_after_reset, is_alive

        print("@daisy_real.reset(): is_alive = {0:s}".format(str(is_alive)))

        # Repeat stabilization until acquired:
        if self.wait4stabilization:
            what2do_next = 0
            while what2do_next == 0:
                state_after_reset[:], what2do_next = self.wait4measurements_stabilization(return_mean=False)
    
        return state_after_reset, is_alive

    def set_status_robot_timeout(self,dt):
        """
        Set a timeout for the socket. When no data received for dt seconds, 
        the socket will raise an exception
        author: alonrot

        :param dt: timeout for the socket, in seconds.
        :return: None
        """
        self.listener_status_robot_socket_process.set_listener_timeout(dt)

    def init_wait4measurements_stabilization(self,cfg_env_comm):

        assert isinstance(cfg_env_comm.time_stabilization,int) or isinstance(cfg_env_comm.time_stabilization,float)
        assert cfg_env_comm.time_stabilization > 0.0
        assert isinstance(cfg_env_comm.freq_acq_stabilization,int)
        assert cfg_env_comm.freq_acq_stabilization > 0.0
        assert isinstance(cfg_env_comm.fac_timeout,int)
        assert cfg_env_comm.fac_timeout > 0
        assert isinstance(cfg_env_comm.tol_stabilization,float)
        self.freq_acq_stabilization = cfg_env_comm.freq_acq_stabilization
        self.Nsteps_stabilization = int(cfg_env_comm.time_stabilization*cfg_env_comm.freq_acq_stabilization)
        # self.state_window_buffer = np.zeros((self.Nsteps_stabilization,len(self.ind_selector)))
        # self.state_window_std = np.zeros(len(self.ind_selector))
        self.state_window_buffer = np.zeros((self.Nsteps_stabilization,self.DIM_STATE))
        self.state_window_std = np.zeros(self.DIM_STATE)
        self.fac_timeout = cfg_env_comm.fac_timeout
        self.tolerance_vec = cfg_env_comm.tol_stabilization*np.ones(self.DIM_STATE)

    def wait4measurements_stabilization(self,return_mean=False):
        """
        
        Simple function that detects when the measurements are stabilized below a threshold.
        Useful to make sure that the filtering of the pose estimation stabilizes before starting the actual experiment

        :param return_mean: 

        NOTE: This function is not senstivie to the case in which we loose track. If we loose track, the Hebi app reports the
        last observed value, which can cause the std to be non zero, and very small. This corner case shoudl be addressed differently:
        Check the is_alive flag, and do not enter this function until the vision signal is working correctly
        """

        # assert self.which_observations == "full", "This function supports base pose stabilization for the moment"

        is_stabilized = False
        Nsteps_timeout = self.fac_timeout*self.Nsteps_stabilization
        ind_stabi = -1
        buffer_is_full = False
        self.state_window_buffer[:] = 0.0
        self.state_window_std[:] = 0.0
        count_timeout = 0
        dt_stabi = 1./self.freq_acq_stabilization
        time_stabi = self.Nsteps_stabilization/self.freq_acq_stabilization
        logging.info("Waiting for pose estimation to stabilize... ({0:2.2f} [sec])".format(time_stabi))
        ind_pose = self.daisy_conf.indices_of["base_pose"] # Get indices of the pose of the base
        while not is_stabilized and count_timeout < Nsteps_timeout:

            ind_stabi = (ind_stabi+1) % self.Nsteps_stabilization

            self.state_window_buffer[ind_stabi,:] = self.read_state()
            if np.all(self.state_window_buffer[ind_stabi,:] == 0.0):
                logging.warning("The observed state is a vector of zeroes (!)")

            if not buffer_is_full and ind_stabi >= self.Nsteps_stabilization-1:
                logging.info("Buffer of {0:2.2f} [sec] filled for the first time...".format(time_stabi))
                buffer_is_full = True

            if buffer_is_full:
                self.state_window_std[:] = np.std(self.state_window_buffer,axis=0)
                if np.all(self.state_window_std[ind_pose] != 0.0) and np.all(self.state_window_std[ind_pose] <= self.tolerance_vec[ind_pose]):
                    is_stabilized = True

            count_timeout += 1

            time.sleep(dt_stabi)

        if return_mean:
            assert buffer_is_full == True
            last_state = np.mean(self.state_window_buffer,axis=0)
        else:
            last_state = self.state_window_buffer[ind_stabi,:]

        if is_stabilized:
            logging.info("The pose estimation is stable. The robot is ready to start a new episode!")
            what2do = 1
        else:
            logging.info("The pose estimation couldn't be stabilized for {0:2.2f} [sec]".format(Nsteps_timeout/self.freq_acq_stabilization))
            print("Chunk of state to be stabilized:",self.state_window_std[ind_pose])
            logging.info("Do you want to try again? [1: No and continue anyways] [0: Yes]")
            ipt = 999
            while not ipt in ["0","1"]:
                ipt = input("Your choice: ")

            what2do = int(ipt)
            if what2do == 0:
                logging.info("[0: Yes, try again]")
            elif what2do == 1:
                logging.info("[1: No and continue]")
            else:
                raise ValueError("Something went really wrong...")

        return last_state, what2do

    def is_robot_still_standing(self):

        base_position_Z = self.state_transformed[self.daisy_conf.indices_trans_of["base_position"]][2]

        still_standing = True
        if base_position_Z < self.base_position_Z_thres:
            log.info("base_position_Z = {0:1.2f} | base_position_Z_thres: {1:1.2f}".format(base_position_Z,self.base_position_Z_thres))
            still_standing = False

        return still_standing

    def is_robot_not_flipped(self):

        sin_ori_X = self.state_transformed[self.daisy_conf.indices_trans_of["base_orientation"][1]]
        sin_ori_Y = self.state_transformed[self.daisy_conf.indices_trans_of["base_orientation"][3]]

        print("sin_ori_X = {0:1.2f} | sin_ori_X_thres: {1:1.2f}".format(sin_ori_X,self.flip_limit))
        print("sin_ori_Y = {0:1.2f} | sin_ori_Y_thres: {1:1.2f}".format(sin_ori_Y,self.flip_limit))

        still_not_flipped = True
        if np.abs(sin_ori_X) > self.flip_limit or np.abs(sin_ori_Y) > self.flip_limit:
            log.info("sin_ori_X = {0:1.2f} | sin_ori_X_thres: {1:1.2f}".format(sin_ori_X,self.flip_limit))
            log.info("sin_ori_Y = {0:1.2f} | sin_ori_Y_thres: {1:1.2f}".format(sin_ori_Y,self.flip_limit))
            still_not_flipped = False

        return still_not_flipped

    def get_state_dim(self):
        # return self.DIM_STATE # Raw state
        return self.DIM_STATE_TRANS # State transformed

    def get_action_dim(self):
        return 18 # TODO: get this number by passing the action dimensionality to the constructor

    def get_state_raw(self,return_tensor=False):
        if return_tensor:
            return torch.from_numpy(self.state_raw).to(self.device)
        else:
            return self.state_raw

    def get_state_transformed(self,return_tensor=False):
        if return_tensor:
            return torch.from_numpy(self.state_transformed).to(self.device)
        else:
            return self.state_transformed

    def get_robot_is_alive_from_status(self,status_robot):
        """
        This function is less light than the others, but we only go through the dictionary when the episode is going to stop anyways
        """
        is_alive = True
        if status_robot < 0:
            logging.error("Robot status reported an error:")
            error_description = self.daisy_conf.get_error_description_from_robot_status(status_robot)
            logging.error(error_description)
            is_alive = False
        return is_alive

    def read_angular_position(self):
        """
        Read angular position from socket
        """
        return self.daisy_interface.get_observations()[self.daisy_conf.indices_of["joint_angular_pos"]]

    def read_state(self):
        """
        Read full state from socket
        """
        return self.daisy_interface.get_observations()

    @staticmethod
    def identity(state):
        return state

    def stop_robot(self):
        self.daisy_interface.stop_robot()

    def get_reward_signal(self,state,action):
        # Reward in distance to desired goal:
        # rew = np.linalg.norm(self.goal_xy - state[self.ind_basepos_xy])
        rew = self.reward_class.get_reward_signal(state,action)
        # import pdb; pdb.set_trace()
        return rew
