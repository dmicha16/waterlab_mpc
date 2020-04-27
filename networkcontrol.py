"""
Module for Level control
"""

import casadi as ca
import mpc


# def mpc_version_1(pump,tank):
def on_off(pump, tank):
    if tank.depth >= tank.full_depth * 0.5:
        pump.target_setting = 5  # [0,1]

    elif tank.depth <= tank.full_depth * 0.1:
        pump.target_setting = 0  # [0,1]

    return pump
