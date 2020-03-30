"""
Module for Level control

"""

import numpy as np
import pandas as pd

# Flag variables for On/Off control 


def on_off(pump, tank):

    flag = 0
    if tank.depth >= tank.full_depth and flag == 0:
        pump.target_setting = 1  # [0,1]
        flag = 1
    elif tank.depth <= tank.full_depth and flag == 1:
        pump.target_setting = 0  # [0,1]
        flag = 0

    return pump