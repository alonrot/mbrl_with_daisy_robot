import numpy as np
DEG2RAD = np.pi / 180.

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
daisy_limits_firmware_min = np.array( [-30., -90.,   0.,
                                       -30.,   0., -90.,
                                       -30., -90.,   0.,
                                       -30.,   0., -90.,
                                       -30., -90.,   0.,
                                       -30.,   0., -90.] ) * DEG2RAD

daisy_limits_firmware_max = np.array( [+30.,   0., +90. ,
                                       +30., +90.,   0. ,
                                       +30.,   0., +90. ,
                                       +30., +90.,   0. ,
                                       +30.,   0., +90. ,
                                       +30., +90.,   0.] ) * DEG2RAD

# freq_lims_global = [0.5, 4.] # Hz
freq_lims_global = [0.1, 2.] # Hz
phase_lims_global = [-np.pi/2, +np.pi/2] # rad

def CPG_limits4PETS():
  """
  NOTE: We need 4 parameters, not 18 again (!!!!)
  """

  Njoints = len(daisy_limits_firmware_min)

  freq_lims = np.array(freq_lims_global)
  phase_lims = np.array(phase_lims_global)
  ampl_lims = np.array([-30., +30.]) * DEG2RAD
  offset_lims = None

  return freq_lims, phase_lims, ampl_lims, offset_lims

def CPG_limits4PETS_as_box():

  # Create box:
  freq_lims, phase_lims, ampl_lims, offset_lims = CPG_limits4PETS()
  if offset_lims is None:
    lim_box_low = np.array([freq_lims[0], phase_lims[0], ampl_lims[0]])
    lim_box_high = np.array([freq_lims[1], phase_lims[1], ampl_lims[1]])
  else:
    lim_box_low = np.array([freq_lims[0], phase_lims[0], ampl_lims[0]], offset_lims[0])
    lim_box_high = np.array([freq_lims[1], phase_lims[1], ampl_lims[1]], offset_lims[1])

  # Create indices dictionary:
  ind_dict = dict(freq=0,phase=1,ampl=2,offset=3)

  return lim_box_low, lim_box_high, ind_dict

def joint_limits4PETS_as_box():

  return daisy_limits_firmware_min, daisy_limits_firmware_max

def joint_limits4PETS_as_box_for_analysis_reward_function(which_ind):

  daisy_limits_firmware_min_aux = -5.*np.ones(18)*DEG2RAD
  daisy_limits_firmware_max_aux = +5.*np.ones(18)*DEG2RAD

  daisy_limits_firmware_min_aux[which_ind] = daisy_limits_firmware_min[which_ind]
  daisy_limits_firmware_max_aux[which_ind] = daisy_limits_firmware_max[which_ind]

  return daisy_limits_firmware_min_aux, daisy_limits_firmware_max_aux


def joint_limits4PETS_as_box_for_walking():

  lim_box_low   = np.array( [ -30., -90., -90.,
                              -30., -90., -90.,
                              -30., -90., -90.,
                              -30., -90., -90.,
                              -30., -90., -90.,
                              -30., -90., -90.] ) * DEG2RAD

  lim_box_high  = np.array( [ +30., +90., +90. ,
                              +30., +90., +90. ,
                              +30., +90., +90. ,
                              +30., +90., +90. ,
                              +30., +90., +90. ,
                              +30., +90., +90.] ) * DEG2RAD

  return lim_box_low, lim_box_high


# # Joint limits compatible with the CPGs, to make the robot stand up:
# def joint_limits4PETS_as_box_for_walking_tight():

#   lim_box_low = np.array( [-30.,   0., -90.,
#                            -30., -30.,   0.,
#                            -30.,   0., -90.,
#                            -30., -30.,   0.,
#                            -30.,   0., -90.,
#                            -30., -30.,   0.] ) * DEG2RAD

#   lim_box_high = np.array( [+30., +30.,   0. ,
#                             +30.,   0., +90. ,
#                             +30., +30.,   0. ,
#                             +30.,   0., +90. ,
#                             +30., +30.,   0. ,
#                             +30.,   0., +90.] ) * DEG2RAD

#   return lim_box_low, lim_box_high

# # Joint limits compatible with the CPGs, to make the robot stand up:
# def joint_limits4PETS_as_box_for_walking_tight():

#   lim_box_low = np.array( [-30., -20., -120.,
#                            -30., -20.,  +60.,
#                            -30., -20., -120.,
#                            -30., -20.,  +60.,
#                            -30., -20., -120.,
#                            -30., -20.,  +60.] ) * DEG2RAD

#   lim_box_high = np.array( [+30., +20.,  -60.,
#                             +30., +20., +120.,
#                             +30., +20.,  -60.,
#                             +30., +20., +120.,
#                             +30., +20.,  -60. ,
#                             +30., +20., +120.] ) * DEG2RAD

#   return lim_box_low, lim_box_high

# Joint limits compatible with the CPGs, to make the robot stand up (same as above, using a less conservative limit in the shoulders)
def joint_limits4PETS_as_box_for_walking_tight():

  lim_box_low = np.array( [-30., -30., -110.,
                           -30., -30.,  +70.,
                           -30., -30., -110.,
                           -30., -30.,  +70.,
                           -30., -30., -110.,
                           -30., -30.,  +70.] ) * DEG2RAD

  lim_box_high = np.array( [+30., +30.,  -70.,
                            +30., +30., +110.,
                            +30., +30.,  -70.,
                            +30., +30., +110.,
                            +30., +30.,  -70. ,
                            +30., +30., +110.] ) * DEG2RAD

  return lim_box_low, lim_box_high


# # Joint limits compatible with the CPGs, to make the robot stand up (same as above, using a less conservative limit in the shoulders)
# def joint_limits4PETS_as_box_for_walking_tight():

#   lim_box_low = np.array( [-30., -30.,  -90.,
#                            -30., -30.,  +60.,
#                            -30., -30.,  -90.,
#                            -30., -30.,  +60.,
#                            -30., -30.,  -90.,
#                            -30., -30.,  +60.] ) * DEG2RAD

#   lim_box_high = np.array( [+30., +30.,  -60.,
#                             +30., +30.,  +90.,
#                             +30., +30.,  -60.,
#                             +30., +30.,  +90.,
#                             +30., +30.,  -60. ,
#                             +30., +30.,  +90.] ) * DEG2RAD

#   return lim_box_low, lim_box_high