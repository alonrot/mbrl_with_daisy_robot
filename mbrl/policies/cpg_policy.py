import os
import sys
import numpy as np
import torch
import logging

from mbrl.optimizers.optimizer import ActionSequence
from tools.tools import get_mbrl_base_path

mbrl_path = get_mbrl_base_path()
sys.path.append("{0:s}/../daisy_toolkit".format(mbrl_path)) # We look for daisy_toolkit in the same folder where mbrl is
from daisy_hardware.cpg_bo import Gait,Robot
import daisy_hardware.cpg as CPG

class ActionSequenceCPG(ActionSequence):

    def __init__(self,offset_shoulder,offset_elbow,action_dim):
        self.offset_shoulder = offset_shoulder # 0.25
        self.offset_elbow = offset_elbow # 1.35
        self.action_dim = action_dim
        # self.freq_action = freq_action

        self.v_curr = None
        self.R_curr = None
        self.tripod = self.define_tripod()

        self.control = CPG.CpgController(self.tripod)
        self.position_desired = np.zeros(action_dim, dtype=np.float64)

        # import pdb; pdb.set_trace()

    def define_tripod(self,v=6,R=0.15):
        """
        GAIT parameters
        Relative phase : 6-by-6 matrix, skew symmetric, phase diff between base joints
        v : frequency of each oscillator [Hz]
        a : constant for each oscillator
        R : amplitude
        amp_offset: constant offset in each oscillator
        phase_offset: offset between base-shoulder and base-elbow joints
        Coupling weights are constant right now - but can be added and changed in CpgController class
        """

        # Input parameters:
        # v = 6 # Hz (new)
        # v = 3 # Hz (original)

        self.v_curr = v
        self.R_curr = R
        logging.info("Policy parameters:")
        logging.info("v: {0:2.2f}".format(self.v_curr))
        logging.info("R: {0:2.2f}".format(self.R_curr))

        # Define tripod:
        daisy_robot = Robot(n_legs=6, n_joint_per_leg=3)
        tripod = Gait(
            robot=daisy_robot,
            relative_phase=[[0,      np.pi,      0,      0,      0,      np.pi],
                            [-np.pi, 0,          np.pi,  0,      0,      0],
                            [0,      -np.pi,     0,      np.pi,  0,      0],
                            [0,      0,          -np.pi, 0,      np.pi,  0],
                            [0,      0,          0,      -np.pi, 0,      np.pi],
                            [-np.pi, 0,          0,      0,      -np.pi, 0]],
            # relative_phase=[[0,     -np.pi,      0,      0,      0,     -np.pi],
            #                 [+np.pi, 0,         -np.pi,  0,      0,      0],
            #                 [0,      +np.pi,     0,     -np.pi,  0,      0],
            #                 [0,      0,          +np.pi, 0,     -np.pi,  0],
            #                 [0,      0,          0,      +np.pi, 0,     -np.pi],
            #                 [+np.pi, 0,          0,      0,      +np.pi, 0]],
            v=[[v for _ in range(6)], [v for _ in range(6)], [v for _ in range(6)]],
            a=[[0.5 for _ in range(6)], [0.5 for _ in range(6)], [0.5 for _ in range(6)]],
            R=[[R for _ in range(6)], [R for _ in range(6)], [R for _ in range(6)]],
            amp_offset=[[0.15 for _ in range(6)], [0.15 for _ in range(6)], [0.15 for _ in range(6)]],
            phase_offset=[[np.pi/2 for _ in range(6)], [np.pi/2 for _ in range(6)]]
        )

        return tripod

    def sample_new_tripod(self):

        # Nominal values:
        v_nom = 6
        R_nom = 0.15

        # New sampled values:
        eps_ = np.random.normal(size=(2,),loc=0.0,scale=0.5)
        v_new = v_nom + eps_[0]
        R_new = R_nom + 0.1*eps_[1]

        self.tripod = self.define_tripod(v=v_new,R=R_new)

    def get_action(self, x, t):
        """
        Override existing function to include the time index
        t: time in seconds
        """

        # if isinstance(t,float):
        #     import pdb; pdb.set_trace()

        # In this case, t must be an index
        if not isinstance(t,int):
            import pdb; pdb.set_trace()
        assert t >= 0
        i = t

        tt = 0
        self.position_desired[:] = 0.0 # Not sure if this is needed
        
        # Update CPG:
        self.control.update()
        for j in range(self.control.n_legs):
            if np.remainder(j, 2) == 0:
                p = int(j / 2)
                self.position_desired[tt] = self.control.y_data[p][i]
                self.position_desired[tt + 1] = -self.offset_shoulder + 2*self.control.y_data[self.control.n_legs + p][i]
                self.position_desired[tt + 2] = -self.offset_elbow + self.control.y_data[2 * self.control.n_legs + p][i]
            else:
                p = int(np.floor(j / 2))
                self.position_desired[tt] = -self.control.y_data[self.control.n_legs - p - 1][i]
                self.position_desired[tt + 1] = self.offset_shoulder - 2*self.control.y_data[2 * self.control.n_legs - p - 1][i]
                self.position_desired[tt + 2] = self.offset_elbow - self.control.y_data[3 * self.control.n_legs - p - 1][i]
            tt += 3

        # import pdb; pdb.set_trace()

        return self.position_desired

    def reset_trajectory(self,resample_tripod=False):
        if resample_tripod:
            self.sample_new_tripod()
        self.control = CPG.CpgController(self.tripod)
        self.position_desired = np.zeros(self.action_dim, dtype=np.float64)

    def get_parameters(self):
        if self.v_curr is None or self.R_curr is None:
            raise ValueError("v_curr, R_curr not initialized ...")
        return dict(v=self.v_curr,R=self.R_curr)


    def __repr__(self):
        return "ActionSequenceCPG"

class CPGPolicy():
    """

    CPG open-loop policy. This class wraps the CPG originally coded in control-daisy/demo_cpg/demo_walking_cpg.py
    which were copied to later created repository daisy_toolkit at daisy_toolkit/daisy_hardware/motion_library.py@demo_walking()
    """
    def __init__(self, device, action_space, planning_horizon,offset_shoulder,offset_elbow,action_dim):
        self.device = device
        self.action_space = action_space
        self.planning_horizon = planning_horizon

        # import pdb; pdb.set_trace()
        self.action_sequence_cpg = ActionSequenceCPG(offset_shoulder,offset_elbow,action_dim)

    def plan_action_sequence(self, state) -> ActionSequence:
        """
        NOTE: There's no planning here as this policy is meant only to collect data to train the dynamics model
        """
        return self.action_sequence_cpg

    def reset_trajectory(self,resample_tripod=False):
        self.action_sequence_cpg.reset_trajectory(resample_tripod)

    def get_parameters(self):
        self.action_sequence_cpg.get_parameters()

