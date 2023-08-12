import numpy as np
import torch

# from mbrl.optimizers import ActionSequence
# from . import Policy
from mbrl.optimizers.optimizer import ActionSequence
# from mbrl.policies.policy import Policy
# from . import Policy


# # Joint limits for moving the robot from standing up position
# daisy_limits_firmware_min = np.array(   [-30., -15., -90.,
#                                          -30., -15., +45.,
#                                          -30., -15., -90.,
#                                          -30., -15., +45.,
#                                          -30., -15., -90.,
#                                          -30., -15., +45] ) * np.pi / 180.

# daisy_limits_firmware_max = np.array(   [+30., +15., -45. ,
#                                          +30., +15., +90. ,
#                                          +30., +15., -45. ,
#                                          +30., +15., +90. ,
#                                          +30., +15., -45. ,
#                                          +30., +15., +90.] ) * np.pi / 180.

# Joint limits for moving the robot from leg-extended, without touching the floor:
daisy_limits_firmware_min = np.array(   [-30., -90.,   0.,
                                         -30.,   0., -90.,
                                         -30., -90.,   0.,
                                         -30.,   0., -90.,
                                         -30., -90.,   0.,
                                         -30.,   0., -90.] ) * np.pi / 180.

daisy_limits_firmware_max = np.array(   [+30.,   0., +90. ,
                                         +30., +90.,   0. ,
                                         +30.,   0., +90. ,
                                         +30., +90.,   0. ,
                                         +30.,   0., +90. ,
                                         +30., +90.,   0.] ) * np.pi / 180.

# freq_lims_global = [0.5, 4.] # Hz
freq_lims_global = [0.1, 2.] # Hz
phase_lims_global = [-np.pi/2, +np.pi/2] # rad

class ActionSequenceSines(ActionSequence):

    def __init__(self, freq_vec, phase_vec, ampl_vec, offset_vec, planning_horizon, device):
        super().__init__()

        # Error checking:
        assert len(freq_vec) == len(phase_vec) == len(ampl_vec) == len(offset_vec)

        self.update_sinewaves_parameters(freq_vec,phase_vec,ampl_vec,offset_vec)

        Njoints = len(freq_vec)
        self.sines = np.zeros(Njoints)
        self.actions = np.zeros(Njoints)

        self.planning_horizon = planning_horizon
        self.sines_plan = np.zeros((self.planning_horizon,Njoints))
        self.actions_plan = np.zeros((self.planning_horizon,Njoints))
        self.t_vec = np.zeros(self.planning_horizon)
        self.device = device

    def update_sinewaves_parameters(self, freq_vec, phase_vec, ampl_vec, offset_vec):
        self.omega_vec = freq_vec*np.pi*2.
        self.phase_vec = phase_vec
        self.ampl_vec = ampl_vec
        self.offset_vec = offset_vec

    def get_action(self, x, t):
        """
        Override existing function to include the time index
        t: time in seconds
        """

        # if isinstance(t,float):
        #     import pdb; pdb.set_trace()

        if not isinstance(t,float) and not isinstance(t,int):
            import pdb; pdb.set_trace()

        self.sines[:] = np.sin(self.omega_vec*t + self.phase_vec)
        self.actions[:] = self.ampl_vec*self.sines + self.offset_vec
        return self.actions

    def generate_action_sequence(self, x, dt, t_curr):

        self.t_vec[:] = np.arange(0,self.planning_horizon)*dt + t_curr

        self.sines_plan[:,:] = np.sin( np.outer(self.t_vec,self.omega_vec) + np.outer(np.ones(self.planning_horizon),self.phase_vec) )
        # import pdb; pdb.set_trace()
        self.actions_plan[:,:] = self.ampl_vec[None,:]*self.sines_plan[:,:] + np.outer(np.ones(self.planning_horizon),self.offset_vec)

        return self.actions_plan

    def get_first_action_from_last_generated_sequence(self):
        return self.actions_plan[0,:]

    def __repr__(self):
        return "ActionSequenceSines"

# class SineWavesPolicy(Policy): # Remove Policy dependency to avoid circular dependency, and python import error
class SineWavesPolicy():
    def __init__(self, device, action_space, planning_horizon):
        self.device = device
        self.action_space = action_space
        self.planning_horizon = planning_horizon

        self.Nactions = len(action_space)
        self.freq_lims = freq_lims_global
        self.phase_lims = phase_lims_global

        self.freq_vec = None
        self.phase_vec = None
        self.ampl_vec = None
        self.offset_vec = None
        self.action_sequence_sines = None

        # Re-sample:
        self.sample_sinewaves_parameters()

        # Action sequence time-dependent:
        self.action_sequence_sines = ActionSequenceSines(self.freq_vec,self.phase_vec,self.ampl_vec,self.offset_vec,self.planning_horizon,self.device)

    def plan_action_sequence(self, state) -> ActionSequence:
        """
        NOTE: There's no planning here as this policy is meant only to collect data to train the dynamics model
        """
        return self.action_sequence_sines

    def reset_trajectory(self):
        pass

    def sample_sinewaves_parameters(self,verbo=True):

        # Generate here the sine waves parameters:
        self.freq_vec = np.random.uniform(low=self.freq_lims[0],high=self.freq_lims[1],size=(self.Nactions,))
        self.phase_vec = np.random.uniform(low=self.phase_lims[0],high=self.phase_lims[1],size=(self.Nactions,))
        self.ampl_vec = np.random.uniform(low=0.0,high=(daisy_limits_firmware_max-daisy_limits_firmware_min)*0.5) # The amplitude is the range / 2
        self.offset_vec = 0.5*(daisy_limits_firmware_min + daisy_limits_firmware_max) # Start in the middle of the range

        # Update the parameters of the waves:
        if self.action_sequence_sines is not None:
            self.action_sequence_sines.update_sinewaves_parameters(self.freq_vec,self.phase_vec,self.ampl_vec,self.offset_vec)

        # TODO alonrot: This is not the way of computing the amplitudes, as we'll get negative numbers,
        # and the waves could be outside the joint limits, when computed from the initial position
        # self.ampl_vec = np.abs(self.ampl_vec)
        if np.any(self.ampl_vec < 0.0):
            raise ValueError("self.ampl_vec cannot be negative...")

        if verbo:
            print("freq_vec")
            print("========")
            print(self.freq_vec)

            print("phase_vec")
            print("========")
            print(self.phase_vec)

            print("ampl_vec")
            print("========")
            print(self.ampl_vec)

    def update_sinewaves_parameters(self, freq_vec, phase_vec, ampl_vec, offset_vec):
        self.freq_vec = freq_vec
        self.phase_vec = phase_vec
        self.ampl_vec = ampl_vec
        self.offset_vec = offset_vec
        self.action_sequence_sines.update_sinewaves_parameters(self.freq_vec,self.phase_vec,self.ampl_vec,self.offset_vec)

    def get_sinewaves_parameters(self):
        my_policy_params = dict(omega_vec=self.freq_vec,phase_vec=self.phase_vec,ampl_vec=self.ampl_vec,offset_vec=self.offset_vec)
        return my_policy_params


class SineWavesPolicyParametrized(SineWavesPolicy):

    def __init__(self, device, parameter_space, planning_horizon, action_size):

        # Dummy object that has __len__ property:
        action_space = [None]*action_size
        super().__init__(device, action_space, planning_horizon)
        self.action_size = action_size

        self.parameter_space = parameter_space
        self.ind_box_dict = self.parameter_space.ind_box_dict

    def update_policy(self,parameters):

        if parameters.ndim > 1:
            parameters = parameters.flatten()

        if not len(self.parameter_space) == len(parameters):
            import pdb; pdb.set_trace() # TODO alonrot: Remove this for future versions

        # Repeat all the aprameters across all the dimensions, for now:
        # TODO alonrot: Improve this
        freq_vec = parameters[self.ind_box_dict["freq"]].item() * np.ones(self.action_size)
        ampl_vec = parameters[self.ind_box_dict["ampl"]].item() * np.ones(self.action_size)
        phase_vec = parameters[self.ind_box_dict["phase"]].item() * np.ones(self.action_size)
        if self.parameter_space == 4:
            offset_vec = parameters[self.ind_box_dict["offset"]].item() * np.ones(self.action_size)
        else:
            offset_vec = np.zeros(self.action_size)

        self.update_sinewaves_parameters(freq_vec,ampl_vec,phase_vec,offset_vec)



        







