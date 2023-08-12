from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import daisy_env as e
from omegaconf import OmegaConf
from gym import spaces

import os

import numpy as np
import torch

# for visualizations
import cv2
import time

# from gym.envs.mujoco import mujoco_env


class DaisyEnv(e.DaisyWalkForward):  # ):

    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.robot_cfg = OmegaConf.from_filename(dir_path + '/assets/daisy_6legs_standing.yaml')
        print("---------- Robot Configuration ----------")
        print(f"Robot Config:\n{self.robot_cfg.pretty()}")
        print("----------------------------------------")

        super(DaisyEnv, self).__init__(**self.robot_cfg)

        ##### ##### ##### ##### ##### ##### ##### #####
        # taken from wrapper and from daisy_env_base
        state, _, _ = self._daisy_base_reset()
        self.old_state = state
        self.old_state['base_pos_global_x'] = state['base_pos_x']
        self.old_state['base_pos_global_y'] = state['base_pos_y']
        for key, val in state.items():
            self._observation_spec[key] = (len(val),)

        # State names for filtering
        # 'j_pos': joint_positions,
        # 'j_vel': joint_velocities,
        # 'j_eff': joint_effort,
        # 'base_pos_x': base_position[0:1],
        # 'base_pos_y': base_position[1:2],
        # 'base_pos_z': base_position[2:],
        # 'base_ori_euler': base_orientation_euler,
        # 'base_ori_quat': base_orientation_quat,
        # 'base_velocity': base_velocity
        # State is j_pos, [0.3752303802828799, -0.6864431979057732, -0.3803587887283957, -0.09377433849112092,
        #                  -0.34270591218509616, -1.1817667247667245, -0.2945693066461283, -0.0006907775445749308,
        #                  -0.34959770227713677, -0.07159166348687875, 0.47408994986084935, 1.7405401458493253,
        #                  0.020269982855398862, 0.4538083083613675, 0.06764172697110439, 0.18097690145770043,
        #                  0.28737105492642634, 1.29017990034541]
        # State is base_pos_x, [0.03452341]
        # State is base_pos_y, [0.05503577]
        # State is base_pos_z, [0.00508331]
        # State is base_ori_euler, (-0.01578792083270051, -0.029523848610948242, -0.4774295353768198)
        # self.state_filter = ['j_eff', 'base_ori_quat']
        self.state_filter = ['j_eff', 'base_ori_quat', 'j_vel', 'base_velocity']
        obs_length = 0
        for key, val in self._observation_spec.items():
            if key in self.state_filter:
                continue
            obs_length += val[0]

        self._obs_length = obs_length
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=[obs_length], dtype=np.float32)
        # TODO The wrapper should actually check the action bounds
        self.action_space = spaces.Box(-1.0, 1.0, shape=[int(len(self.robot.ordered_joints))], dtype=np.float32)
        self.reward_func = None

        # TODO: I think this isn't needed here
        # TODO alonrot: Defined a new class member, which will get rewrittena t each iteration
        # TODO alonrot: The state dimension needs to be read from daisy.yaml somehow
        self.state_vec = np.zeros(24) # Use here env.state_size
        self.ind_state_vec_dict = dict(joint_angular_pos=np.arange(0,18),
                                        base_position=np.arange(18,21),
                                        base_orientation=np.arange(21,24))

        # import pdb;pdb.set_trace()

    def render(self):
        self._env.render()

    def step(self, a):

        # TODO alonrot: a -> action, nunmpy array
        state, reward, done = super().step(a)  # Which _step() is being called?


        """Applies action a and returns next state.

        Args:
            a (list or numpy array): List of actions. Lenght must be the same as
                get_action_spec() defines.

        Returns:
            state (dict): The dict contains four different states. 'j_pos' are the
                joint positions. 'j_vel' are the current velocities of the
                joint angles. 'base_pos' is the 3D position of the base of Daisy
                in the world. 'base_ori_euler' is the orientation of the robot
                in euler angles.
            reward (float): Reward achieved.
            done (bool): True if end of simulation reached. (Currently always False)
        """

        self.original_state = state # TODO alonrot: Where is this further used?

        # TODO: This filters out the undesired measurements, defined in self.state_filter, and
        # TODO leaves only j_pos, base_pos, base_ori_euler (18+3+3)
        # TODO The state length should agree with the parameter specified in pets/conf/env/daisy.yaml
        state_conc = []
        self.index_dict = {}
        for key, val in state.items():
            if key in self.state_filter:
                continue
            self.index_dict[key] = len(state_conc)
            state_conc.extend(val) # TODO Why extend?
            print(f"State is {key}, {val}") # TODO alonrot: printing out state

        # TODO: Compute reward - We recompute the reward here, instead of
        # TODO  using the one above (the one from daisy_env)
        state = np.array(state_conc)

        # TODO replace the above code by this, which does not have any dynamic allocation:
        # for key, val in state.items():
        #     self.state_vec[self.ind_state_vec_dict[key]] = val

        reward = self.reward_func(state, 0)

        print('state dimension is', np.shape(state))

        return state, reward, bool (done), {}

    def reset(self):
        state, reward, done = super().reset()
        self.original_state = state
        state_conc = []
        self.index_dict = {}
        for key, val in state.items():
            if key in self.state_filter:
                continue
            self.index_dict[key] = len(state_conc)
            state_conc.extend(val)

        state = np.array(state_conc)
        return state

    @staticmethod
    def get_reward_1(next_ob, _action):
        """
        :param next_ob: the new state we got to
        :param action:  the action that led to this state
        :return: the reward for the transition

        Next ob is going to be a tensor of size (batch x dim)
        .. for a state_filter of  ['j_vel', 'j_eff', 'base_ori_quat', 'base_velocity'] you will have a list of
        (not 100% confirmed)

        00, 01, 02 : ['base1', 'shoulder1', 'elbow1',
        03, 04, 05 : 'base2', 'shoulder2', 'elbow2',
        06, 07, 08 : 'base3', 'shoulder3', 'elbow3',
        09, 10, 11 : 'base4', 'shoulder4', 'elbow4',
        12, 13, 14 : 'base5', 'shoulder5', 'elbow5',
        15, 16, 17 : 'base6', 'shoulder6', 'elbow6',
        18, 19, 20 : base_pos_x, base_pos_y, base_pos_z
        21, 22, 23 : base_ori_euler[0], base_ori_euler[1], base_ori_euler[2]]
        """

        # reward = torch.zeros((next_ob.shape[0], 1))

        # hand design reward function
        # for key, val in next_ob.items():
        #     if key in next_ob.state_filter:
        #         continue
        #
        #     # Can add x pos (walk forward)
        #     if key == 'base_pos_x':
        #         reward += val
        #         raise NotImplementedError("Do something for this state")
        # print(f"State is {key}, {val}")
        import pdb; pdb.set_trace()
        reward = next_ob[:, 18] + next_ob[:, 19]
        return reward

    @staticmethod
    def get_reward_2(next_ob, action):
        """
        :param next_ob: the new state we got to
        :param action:  the action that led to this state
        :return: the reward for the transition

        Next ob is going to be a tensor of size (batch x dim)
        .. for a state_filter of  ['j_vel', 'j_eff', 'base_ori_quat', 'base_pos_x', 'base_pos_y','base_pos_z']
           you will have a list of
           (not 100% confirmed)

        00, 01, 02 : ['base1', 'shoulder1', 'elbow1',
        03, 04, 05 : 'base2', 'shoulder2', 'elbow2',
        06, 07, 08 : 'base3', 'shoulder3', 'elbow3',
        09, 10, 11 : 'base4', 'shoulder4', 'elbow4',
        12, 13, 14 : 'base5', 'shoulder5', 'elbow5',
        15, 16, 17 : 'base6', 'shoulder6', 'elbow6',
        18, 19, 20 : base_ori_euler[0], base_ori_euler[1], base_ori_euler[2]],
        21 : 'base_velocity'

        Reward is simply maximizing velocity
        """

        # reward = torch.zeros((next_ob.shape[0], 1))

        # hand design reward function
        # for key, val in next_ob.items():
        #     if key in next_ob.state_filter:
        #         continue
        #
        #     # Can add x pos (walk forward)
        #     if key == 'base_pos_x':
        #         reward += val
        #         raise NotImplementedError("Do something for this state")
        # print(f"State is {key}, {val}")
        reward = next_ob[:, -2]
        return reward

    @staticmethod
    def get_reward_3(next_ob, action):
        """
        :param next_ob: the new state we got to
        :param action:  the action that led to this state
        :return: the reward for the transition

        Next ob is going to be a tensor of size (batch x dim)
        .. for a state_filter of  ['j_vel', 'j_eff', 'base_ori_quat', 'base_pos_x', 'base_pos_y','base_pos_z']
           you will have a list of
           (not 100% confirmed)

        00, 01, 02 : ['base1', 'shoulder1', 'elbow1',
        03, 04, 05 : 'base2', 'shoulder2', 'elbow2',
        06, 07, 08 : 'base3', 'shoulder3', 'elbow3',
        09, 10, 11 : 'base4', 'shoulder4', 'elbow4',
        12, 13, 14 : 'base5', 'shoulder5', 'elbow5',
        15, 16, 17 : 'base6', 'shoulder6', 'elbow6',
        18, 19, 20 : base_ori_euler[0], base_ori_euler[1], base_ori_euler[2]],
        21 : 'base_velocity'

        Reward is simply maximizing velocity with a squared cost on pitch and roll angles.
        """

        # reward = torch.zeros((next_ob.shape[0], 1))

        # hand design reward function
        # for key, val in next_ob.items():
        #     if key in next_ob.state_filter:
        #         continue
        #
        #     # Can add x pos (walk forward)
        #     if key == 'base_pos_x':
        #         reward += val
        #         raise NotImplementedError("Do something for this state")
        # print(f"State is {key}, {val}")
        reward = next_ob[:, -2]
        reward -= .05*(torch.pow(next_ob[:, 18],2)+torch.pow(next_ob[:, 19],2))

        return reward

    @staticmethod
    def get_reward_4(next_ob, action):
        """
        :param next_ob: the new state we got to
        :param action:  the action that led to this state
        :return: the reward for the transition

        Created by Akshara for BO_SVAE_DC exps

        Next ob is going to be a tensor of size (batch x dim)
        .. for a state_filter of  ['j_eff', 'base_ori_quat', 'base_velocity', 'j_pos']
           you will have a list of


        00, 01, 02 : ['base1_vel', 'shoulder1_vel', 'elbow1',
        03, 04, 05 : 'base2', 'shoulder2', 'elbow2',
        06, 07, 08 : 'base3', 'shoulder3', 'elbow3',
        09, 10, 11 : 'base4', 'shoulder4', 'elbow4',
        12, 13, 14 : 'base5', 'shoulder5', 'elbow5',
        15, 16, 17 : 'base6', 'shoulder6', 'elbow6',
        18, 19, 20 : base_pos_x, base_pos_y, base_pos_z,
        21, 22, 23: base_ori_euler[0], base_ori_euler[1], base_ori_euler[2]]
        ],


        Reward is simply maximizing forward displacement and penalizing high joint velocities
        """

        # hand design reward function
        # for key, val in next_ob.items():
        #     if key in next_ob.state_filter:
        #         continue
        #
        #     # Can add x pos (walk forward)
        #     if key == 'base_pos_x':
        #         reward += val
        #         raise NotImplementedError("Do something for this state")
        # print(f"State is {key}, {val}")
        reward = 10.0*next_ob[:,19]
        MAX_JOINT_VEL_SIM = 5.0
        # MAX_X = 1.0
        j_vel = next_ob[:, :18]
        j_max = torch.max(j_vel, 1).values
        t = j_max > MAX_JOINT_VEL_SIM
        t = t.to(dtype=torch.float)

        reward = reward - t

        return reward

    @staticmethod
    def get_reward_5(next_ob, action):
        """
        :param next_ob: the new state we got to
        :param action:  the action that led to this state
        :return: the reward for the transition

        Created by Akshara for MBRL exps

        Next ob is going to be a tensor of size (batch x dim)
        .. for a state_filter of  ['j_eff', 'base_ori_quat', 'j_vel']
           you will have a list of


        00, 01, 02 : ['base1', 'shoulder1', 'elbow1',
        03, 04, 05 : 'base2', 'shoulder2', 'elbow2',
        06, 07, 08 : 'base3', 'shoulder3', 'elbow3',
        09, 10, 11 : 'base4', 'shoulder4', 'elbow4',
        12, 13, 14 : 'base5', 'shoulder5', 'elbow5',
        15, 16, 17 : 'base6', 'shoulder6', 'elbow6',
        18, 19, 20 : ['base1', 'shoulder1', 'elbow1',
        21, 22, 23 : 'base2', 'shoulder2', 'elbow2',
        24, 25, 26 : 'base3', 'shoulder3', 'elbow3',
        27, 28, 29 : 'base4', 'shoulder4', 'elbow4',
        30, 31, 32 : 'base5', 'shoulder5', 'elbow5',
        33, 34, 35 : 'base6', 'shoulder6', 'elbow6',
        36, 37, 38 : base_pos_x, base_pos_y, base_pos_z,
        39, 40, 41: base_ori_euler[0], base_ori_euler[1], base_ori_euler[2]]
        ],


        Reward is trying to reach a goal position
        """

        # TODO alonrot: Here we set the goal, hardcoded

        if torch.is_tensor(next_ob):
            goal = torch.tensor([5,5], device=next_ob.device, dtype=torch.float)
            robot_pos = next_ob[:,18:20]
            reward = -2.0* torch.norm(robot_pos-goal, dim=1)
        else:
            # print('reward 5 was called')
            goal = np.array([5,5])
            robot_pos = np.array(next_ob[18:20])
            reward = -2.0 * np.linalg.norm(robot_pos-goal)

        return reward

    @staticmethod
    def get_reward_6(next_ob, action):

        if torch.is_tensor(next_ob):
            goal = torch.tensor([-5, 5], device=next_ob.device, dtype=torch.float)
            robot_pos = next_ob[:, 18:20]
            reward = -2.0 * torch.norm(robot_pos - goal, dim=1)
        else:
            # print('reward 6 was called')
            goal = np.array([-5, 5])
            robot_pos = np.array(next_ob[18:20])
            reward = -2.0 * np.linalg.norm(robot_pos - goal)
        return reward

    @staticmethod
    def get_reward_7(next_ob, action):

        if torch.is_tensor(next_ob):
            goal = torch.tensor([0, 5], device=next_ob.device, dtype=torch.float)
            robot_pos = next_ob[:, 18:20]
            reward = -2.0 * torch.norm(robot_pos - goal, dim=1)
        else:
            # print('reward 7 was called')
            goal = np.array([0, 5])
            robot_pos = np.array(next_ob[18:20])
            reward = -2.0 * np.linalg.norm(robot_pos - goal)
        return reward

    @staticmethod
    def get_reward_8(next_ob, action):

        if torch.is_tensor(next_ob):
            goal = torch.tensor([5, 0], device=next_ob.device, dtype=torch.float)
            robot_pos = next_ob[:, 18:20]
            reward = -2.0 * torch.norm(robot_pos - goal, dim=1)
        else:
            # print('reward 8 was called')
            goal = np.array([5, 0])
            robot_pos = np.array(next_ob[18:20])
            reward = -2.0 * np.linalg.norm(robot_pos - goal)
        return reward

    @staticmethod
    def get_reward_9(next_ob, action):

        if torch.is_tensor(next_ob):
            goal = torch.tensor([-5, 0], device=next_ob.device, dtype=torch.float)
            robot_pos = next_ob[:, 18:20]
            reward = -2.0 * torch.norm(robot_pos - goal, dim=1)
        else:
            # print('reward 8 was called')
            goal = np.array([-5, 0])
            robot_pos = np.array(next_ob[18:20])
            reward = -2.0 * np.linalg.norm(robot_pos - goal)
        return reward

    @staticmethod
    def preprocess_state(state):
        assert torch.is_tensor(state)
        assert state.dim() in (1, 2)

        raise NotImplementedError("Need to figure out built in normalization of data")
        return ret

    @staticmethod
    def identity(state):
        return state


class EnvironmentHook(object):
    def __init__(self):
        pass

    def step(self, env, state, reward, done):
        pass

    def reset(self, env, state, reward, done):
        pass


class SimpleVideoRecorder(EnvironmentHook):
    def __init__(self, name='Daisy', path=None):
        if path is None:
            self._path = '/tmp'
        else:
            self._path = path
        self.name = name
        self._counter = 0
        self._frame_width = 320
        self._frame_height = 200
        self._create_vid_stream()
        self._fps_per_frame = 0
        self._number_of_steps = 1

    def gen_from_ep(self, env, ep):
        """
        Generates a video in /tmp/ from a passed episode (loaded from checkpoint)
        :param env: env gym object for printing frames
        :param ep: episode loaded from data
        :return: prints path where video is saved
        """
        print("Processing video frame from source ep")
        # create vid stream
        self._create_vid_stream()

        # Write first frame
        frame = self.write_frame(env)
        self._out.write(frame)

        # reest state
        env.reset()

        state0 = ep[0].s0
        for i, SAS in enumerate(ep):
            print(f"--frame {i} done.")
            # state = SAS.s0
            action = SAS.a

            state, reward, _, __ = env.step(action.cpu().detach().numpy())
            frame = self.write_frame(env)
            self._out.write(frame)

        quit()

    def write_frame(self, env):
        frame = env.render_camera_image((self._frame_width, self._frame_height))
        frame = frame * 255
        frame = frame.astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def _create_vid_stream(self):
        print("Saving to {}-video_{}.avi".format(self.name, self._counter))
        self._out = cv2.VideoWriter(os.path.join(self._path, '{}-video_{}.avi'.format(self.name, self._counter)),
                                    cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                    (self._frame_width, self._frame_height))


