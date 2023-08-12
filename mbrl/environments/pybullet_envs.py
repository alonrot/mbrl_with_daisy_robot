import os
import numpy as np
from gym import error, spaces
from gym.utils import seeding
import gym

try:
    import pybullet
    import pybullet_data
    from pybullet_utils.bullet_client import BulletClient
except Exception as e:
    raise error.DependencyNotInstalled("{}. HINT: You need to install PyBullet, i.e. pip install pybullet )".format(e))

dir_path = os.path.dirname(os.path.realpath(__file__))


class PyBulletEnv(gym.Env):
    """
    Superclass for all PyBullet environments.
    """

    def __init__(self, relative_model_path, gui, controlled_joints, ee_idx, rest_pose=None, torque_limits=None):
        if gui:
            self._sim = BulletClient(connection_mode=pybullet.GUI)
        else:
            self._sim = BulletClient(connection_mode=pybullet.DIRECT)
        self._sim.setAdditionalSearchPath(pybullet_data.getDataPath())

        _, ext = os.path.splitext(relative_model_path)
        if ext == '.urdf':
            self._robot_id = self._sim.loadURDF(f'{dir_path}/assets/{relative_model_path}', basePosition=[-0.5, 0, 0.0],
                                                useFixedBase=True)
        elif ext == '.xml':
            _, self._robot_id = self._sim.loadMJCF(f'{dir_path}/assets/{relative_model_path}')
        else:
            raise Exception('Unknown extension.')

        self._controlled_joints = controlled_joints
        self._ee_idx = ee_idx
        self._n_dofs = len(controlled_joints)
        if rest_pose is None:
            self._rest_pose = np.zeros(self._n_dofs * 2)
        else:
            self._rest_pose = rest_pose
        self._plane_id = pybullet.loadURDF("plane.urdf", [0, 0, 0])

        if torque_limits is None:
            torque_limits = np.zeros(self._n_dofs)
        self.action_space = spaces.Box(low=-torque_limits, high=torque_limits, dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2 * self._n_dofs,), dtype=np.float32)

        self._sim.setGravity(0, 0, -9.81)
        hz = 240
        self.dt = 1.0 / hz
        self._sim.setTimeStep(self.dt)
        self._sim.setRealTimeSimulation(0)
        self._sim.setJointMotorControlArray(self._robot_id,
                                            self._controlled_joints,
                                            pybullet.VELOCITY_CONTROL,
                                            forces=np.zeros(self._n_dofs))

    def disconnect(self):
        self._sim.disconnect()

    def inverse_kinematics(self, target_position):
        """
        :param target_position: Target end-effector position in Cartesian coordinates.
        :return des_joint_state: Joint state which achieves end-effector position, computed from PyBullet.
        """
        des_joint_state = self._sim.calculateInverseKinematics(self._robot_id,
                                                               self._ee_idx,
                                                               target_position,
                                                               jointDamping=[0.1 for _ in range(7)])
        return np.asarray(des_joint_state)

    def reset(self, state=None):
        """
        :param state: Numpy array of shape (self._n_dofs * 2,), representing n_dofs of joint positions and
                      n_dofs of joint velocities. PyBullet resets to this state.
        :return: state after one simulation step.
        """
        if state is None:
            state = self._rest_pose
        assert state.shape == (self._n_dofs * 2,)
        joint_pos = state[:self._n_dofs]
        joint_vel = state[self._n_dofs:]

        for i in range(self._n_dofs):
            self._sim.resetJointState(bodyUniqueId=self._robot_id,
                                      jointIndex=self._controlled_joints[i],
                                      targetValue=joint_pos[i],
                                      targetVelocity=joint_vel[i])

        self._sim.stepSimulation()
        return self.get_state()

    def step(self, action):
        """
        :param action: Numpy array of shape (self._n_dofs,) representing n_dofs of joint torques.
        :return: Default OpenAI Gym Tuple (obs, reward=0, done=False, info={})
        """
        self._sim.setJointMotorControlArray(bodyIndex=self._robot_id,
                                            jointIndices=self._controlled_joints,
                                            controlMode=pybullet.TORQUE_CONTROL,
                                            forces=action)
        self._sim.stepSimulation()
        ob = self.get_state()
        reward = 0
        done = False
        return ob, reward, done, {}

    def reset_then_step(self, state, action):
        self.reset(state)
        ob, _reward, _done, _ = self.step(action)
        return ob

    def get_state(self):
        cur_joint_states = self._sim.getJointStates(self._robot_id, self._controlled_joints)
        cur_joint_angles = [cur_joint_states[i][0] for i in range(self._n_dofs)]
        cur_joint_vel = [cur_joint_states[i][1] for i in range(self._n_dofs)]
        return np.hstack([cur_joint_angles, cur_joint_vel])

    def get_joint_position(self):
        return self.get_state()[:self._n_dofs]

    def get_joint_velocity(self):
        return self.get_state()[self._n_dofs:]

    def get_ee_position(self):
        """
        :return: Cartesian end-effector position (Numpy array of size 3).
        """
        ee_state = self._sim.getLinkState(self._robot_id, self._ee_idx)
        return np.array(ee_state[0])

    def get_ee_velocity(self):
        """
        :return: Cartesian end-effector linear and angular velocity (2-tuple of Numpy arrays of size 3).
        """
        ee_state = self._sim.getLinkState(self._robot_id, self._ee_idx, computeLinkVelocity=1)
        linear_velocity, angular_velocity = np.array(ee_state[6]), np.array(ee_state[7])
        return linear_velocity, angular_velocity

    def get_ee_jacobian(self):
        """
        :return: 2 tuple of (linear Jacobian, angular Jacobian) computed by PyBullet, as Numpy arrays.
        """
        cur_joint_states = self._sim.getJointStates(self._robot_id, self._controlled_joints)
        cur_joint_angles = [cur_joint_states[i][0] for i in range(self._n_dofs)]
        cur_joint_vel = [cur_joint_states[i][1] for i in range(self._n_dofs)]

        bullet_jac_lin, bullet_jac_ang = self._sim.calculateJacobian(
            bodyUniqueId=self._robot_id,
            linkIndex=self._ee_idx,
            localPosition=[0, 0, 0],
            objPositions=cur_joint_angles,
            objVelocities=cur_joint_vel,
            objAccelerations=[0] * self._n_dofs,
        )
        return np.asarray(bullet_jac_lin), np.asarray(bullet_jac_ang)

    def inverse_dynamics(self, desired_acceleration):
        """
        :param desired_acceleration: List of size n_dofs.
        :return: Numpy array of torques needed to achieve desired acceleration, computed from PyBullet.
        """
        cur_joint_states = self._sim.getJointStates(self._robot_id, self._controlled_joints)
        cur_joint_angles = [cur_joint_states[i][0] for i in range(self._n_dofs)]
        cur_joint_vel = [cur_joint_states[i][1] for i in range(self._n_dofs)]
        torques = self._sim.calculateInverseDynamics(self._robot_id,
                                                     cur_joint_angles,
                                                     cur_joint_vel,
                                                     desired_acceleration)
        return np.asarray(torques)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def compute_next_state(self, state, action):
        return self.reset_then_step(state, action)

    def compute_next_states(self, states, actions):
        new_states = np.full_like(states, -1)
        for i in range(len(states)):
            new_states[i] = self.compute_next_state(states[i], actions[i])
        return new_states


class PyBulletReacher(PyBulletEnv):
    def __init__(self, gui=False):
        rel_model_path = "reacher.xml"
        ee_idx = 4
        controlled_joints = [0, 2]
        torque_limits = np.array([1, 1])

        super(PyBulletReacher, self).__init__(relative_model_path=rel_model_path,
                                              gui=gui,
                                              controlled_joints=controlled_joints,
                                              ee_idx=ee_idx,
                                              torque_limits=torque_limits,
                                              )


class PyBulletKuka(PyBulletEnv):
    def __init__(self, gui=False):
        rel_model_path = 'kuka/iiwa7.urdf'
        ee_idx = 7
        controlled_joints = range(7)
        rest_pose = [0.0, 0.0, 0.0, 1.57079633, 0.0, 1.03672558, 0.0]
        torque_limits = np.array([80, 80, 40, 40, 9, 9, 9])

        super(PyBulletKuka, self).__init__(relative_model_path=rel_model_path,
                                           gui=gui,
                                           controlled_joints=controlled_joints,
                                           ee_idx=ee_idx,
                                           rest_pose=rest_pose,
                                           torque_limits=torque_limits,
                                           )


class PyBulletSawyer(PyBulletEnv):
    def __init__(self, gui=False):
        rel_model_path = 'sawyer/sawyer_bullet.urdf'
        ee_idx = 17
        controlled_joints = [3, 8, 9, 10, 11, 13, 16]
        rest_pose = [-0., -1.18, 0., 2.18, -0., 0.57, 3.14]
        torque_limits = np.array([200, 200, 100, 100, 100, 30, 30])

        super(PyBulletSawyer, self).__init__(relative_model_path=rel_model_path,
                                             gui=gui,
                                             controlled_joints=controlled_joints,
                                             ee_idx=ee_idx,
                                             rest_pose=rest_pose,
                                             torque_limits=torque_limits,
                                             )
