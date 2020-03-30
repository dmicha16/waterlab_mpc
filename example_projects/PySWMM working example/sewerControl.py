"""
Module for Level control

"""

import numpy as np

# Flag variables for On/Off control 
flag = [0,0];

def on_off(tanks,pumps,tank_maxdepth,tank_mindepth):

    if tanks[0].depth >= np.array(tank_maxdepth)[0] and flag[0] == 0:
        pumps[0].target_setting = 0.6  # [0,1]
        flag[0] = 1
    elif tanks[0].depth <= np.array(tank_mindepth)[0] and flag[0] == 1:
        pumps[0].target_setting = 0  # [0,1]
        flag[0] = 0
        
    if tanks[1].depth >= np.array(tank_maxdepth)[1] and flag[1] == 0:
        pumps[1].target_setting = 1  # [0,1]
        flag[1] = 1
    elif tanks[1].depth <= np.array(tank_mindepth)[1] and flag[1] == 1:
        pumps[1].target_setting = 0  # [0,1]
        flag[1] = 0