#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 20:12:58 2020

@author: davidm
"""

import matplotlib.pyplot as plt

import os

import pandas as pd
import casadi as ca
import numpy as np

import time

from datetime import datetime
from pyswmm import Simulation, Nodes, Links
import networkcontrol as controller
from controller import mpc, mpco
# import plotter as plotter
from enum import Enum

from state_space_models import euler_model,\
    custom_model, preismann_model


class SimType(Enum):
    EULER = 1
    PREISMANN = 2
    CUSTOM_MODEL = 3


def print_welcome_msg():
    """
    Prints a welcome msg and general status of the project.
    :return: None
    """
    # TODO: add ascii art ;)
    # TODO: add perhaps version of the code base?

    print("A python project of a Model Predictive Controller (MPC)"
          " for Urban Drainage Networks (UDNs) to mitigate Combined Sewer Overflows (CSOs)")


def define_state_space_model(simulation_type, pred_horizon, disturb_magnitude):

    if simulation_type == SimType.EULER:
        initial_state_space_model = euler_model.make_euler_model(simulation_type, pred_horizon,
                                                                 disturb_magnitude)

    elif simulation_type == SimType.PREISMANN:
        initial_state_space_model = preismann_model.make_preismann_model(simulation_type, pred_horizon,
                                                                         disturb_magnitude)

    elif simulation_type == SimType.CUSTOM_MODEL:
        initial_state_space_model = custom_model.make_custom_model_model(simulation_type, pred_horizon,
                                                                         disturb_magnitude)

    else:
        print("Default, going with generic model.")
        initial_state_space_model = custom_model.make_custom_model_model(simulation_type, pred_horizon,
                                                                         disturb_magnitude)

    return initial_state_space_model


def set_init_conditions_weight_matrices(pred_model, pred_horizon, ctrl_horizon, disturb_magnitude):

    Ap = pred_model["system_model"]
    Bp = pred_model["b_matrix"]
    Bp_d = pred_model["b_disturbance"]

    states = pred_model["num_states"]
    inputs = pred_model["num_inputs"]

    # Q, R is required for the model and it's defined here
    Q = ca.DM([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    R = ca.DM([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])

    # To run the simulation you need the previous control input, in this case since the simulation is not run yet,
    # these numbers are arbitrary
    # initial conditions, specific for the model
    x0 = ca.DM([[10], [11], [12]])
    u0 = ca.DM([-1, -3, -6])

    ref = set_reference(prediction_horizon, states)

    # create mpc_model and construct the state_space_model
    mpc_model = mpco.MpcObj(Ap, Bp, control_horizon, prediction_horizon, Q, R, x0, u0, ref, Bp_d, disturbance_matrix)

    state_space_model = {"system_model": A, "B_matrix": B, "B_d": B_d, "disturbance": disturbance_matrix,
                         "disturbance_magnitude": disturbance_magnitude, "Q_matrix": Q, "R_matrix": R,
                         "inputs": inputs, "states": states, "reference": ref, "x0": x0, "u0": u0}

    return [mpc_model, state_space_model]


def set_reference(prediction_horizon, states):
    """
    Create an arbitrary reference
    # TODO: make sure that this is the right reference
    :param prediction_horizon:
    :param states:
    :return:
    """

    ref = ca.DM.ones(prediction_horizon * states, 1)
    for state in range(states):
        ref[state::states] = ref[state::states] + state - 2

    return ref


def run_simulation(time_step, pumps, tanks, network_df, network_name, complete_model):

    print("Running Simulation!")
    time.sleep(1)

    mpc_model = complete_model[0]
    state_space_model = complete_model[1]

    A = state_space_model["system_model"]
    B = state_space_model["B_matrix"]
    B_d = state_space_model["B_d"]
    inputs = state_space_model["inputs"]
    states = state_space_model["states"]
    disturbance_magnitude = state_space_model["disturbance_magnitude"]
    disturbance = state_space_model["disturbance"]
    ref = state_space_model["reference"]

    x = state_space_model["x0"]
    u = state_space_model["u0"]

    steps_between_plots = 3

    plot_mpc_steps = True

    with Simulation(network_name) as sim:

        # pump1 = Links(sim)[pumps[0]]
        # pump2 = Links(sim)[pumps[1]]
        # tank1 = Nodes(sim)[tanks[0]]
        # tank2 = Nodes(sim)[tanks[1]]

        sim.step_advance(time_step)

        user_key_input = input("press s key to step, or \'r\' to run all steps, or q to quit")

        for idx, step in enumerate(sim):

            u = u + mpc_model.get_next_control_input_change()
            x = A @ x + B @ u + B_d @ disturbance[:inputs]
            new_disturbance = (np.random.rand(inputs) - 0.5) * disturbance_magnitude / 2
            disturbance = np.append(disturbance[inputs:], new_disturbance)

            # TODO: finish filling of dataframe
            # network_df = network_df.append(pd.Series([tank1.volume, tank1.depth, tank1.flooding, tank1.total_inflow,
            #                                           pump1.flow], index=network_df.columns), ignore_index=True)

            if user_key_input == "r":
                mpc_model.plot_progress(options={'drawU': 'U'}, ignore_inputs=[1, 2])
                mpc_model.step(x, u, ref, disturbance)
                print(f"Progress {int(sim.percent_complete * 100)}%", end="\r")

            elif user_key_input == "s":
                mpc_model.plot_progress(options={'drawU': 'U'}, ignore_inputs=[1, 2])
                mpc_model.plot_step({'drawU': 'both'})
                user_key_input = input("press s key to step, or \'r\' to run all steps, or q to quit")

            elif user_key_input == "q":
                print("Quitting.")
                break

    return network_df


def save_data(simulation_df):

    today = datetime.now()
    timestamp_day = today.strftime("%m_%d_%y")
    timestamp_hour = today.strftime("%H_%M_%S")
    path = f"data/mpc_data/{timestamp_day}"

    directory_contents = os.listdir("data/mpc_data/")
    if timestamp_day not in directory_contents:
        os.mkdir(path)

    file_name = f"{path}/mpc_simulation_df_{timestamp_hour}.pkl"

    print(file_name)

    simulation_df.to_pickle(file_name)


if __name__ == "__main__":

    print_welcome_msg()

    # create columns and pandas DataFrame to store data
    columns = ['timestamp', 'time_step', 'tank_volume', 'tank_depth', 'tank_overflow', 'tank_inflow', 'pump_flow']
    network_df = pd.DataFrame(columns=columns)

    network_name = "epa_networks/project_network/project_network.inp"

    # step size [seconds]
    time_step = 30

    pumps = ["P1", "P2"]
    tanks = ["Tank1", "Tank2"]

    prediction_horizon = 40
    control_horizon = 40

    disturbance_magnitude = 5

    prediction_model = define_prediction_model()

    complete_model = define_real_model(prediction_model, prediction_horizon, control_horizon, disturbance_magnitude)

    simulation_df = run_simulation(time_step, pumps, tanks, network_df, network_name, complete_model)
    # simulation_df = network_df

    save_data(simulation_df)







