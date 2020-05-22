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
from pyswmm import Simulation
from util_scripts import disturbance_reader
import networkcontrol as controller
from controller import mpc, mpco
# import plotter as plotter
from enum import Enum

from state_space_models import euler_model, \
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
    """
    Interface to the specific model declarations. The appropriate model construction is called
    specified by the simulation type passed to this function.
    :param simulation_type: Enum to switch between model types.
    :param pred_horizon: Length of the prediction_horizon.
    :param disturb_magnitude: Magnitude of the disturbance.
    :return: The constructed initial state space model.
    """

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


def make_mpc_model(ss_model, pred_horizon, ctrl_horizon):
    """
    Create a model specific MPC
    :param ss_model: Model specific state space representation
    :param pred_horizon: Length of prediction horizon
    :param ctrl_horizon: Length of control horizon
    :return: An augmented state space model with a new entry in the dict with the MPC object
    """

    simulation_type = ss_model["sim_type"]

    if simulation_type == SimType.EULER:
        aug_state_space_model = euler_model.make_euler_mpc_model(ss_model, pred_horizon,
                                                                 ctrl_horizon)
    elif simulation_type == SimType.PREISMANN:
        aug_state_space_model = preismann_model.make_preismann_mpc_model(ss_model, pred_horizon,
                                                                         ctrl_horizon)
    elif simulation_type == SimType.CUSTOM_MODEL:
        aug_state_space_model = preismann_model.make_preismann_mpc_model(ss_model, pred_horizon,
                                                                         ctrl_horizon)
    else:
        aug_state_space_model = preismann_model.make_preismann_mpc_model(ss_model, pred_horizon,
                                                                         ctrl_horizon)

    return aug_state_space_model


# Functionality moved to different files
# def set_reference(pred_horizon, states):
#     """
#     Create an arbitrary reference
#     # TODO: make sure that this is the right reference
#     :param pred_horizon:
#     :param states:
#     :return:
#     """
#
#     ref = ca.DM.ones(pred_horizon * states, 1)
#     for state in range(states):
#         ref[state::states] = ref[state::states] + state - 2
#
#     return ref


def run_simulation(time_step, pump_ids, tank_ids, junction_ids, network_df, network_path_name, complete_model,
                   steps_between_plots, plot_mpc_steps):
    print("Running Simulation!")
    time.sleep(1)

    mpc_model = complete_model["mpc_model"]

    if state_space_model["sim_type"] == SimType.CUSTOM_MODEL:
        custom_model.run_custom_model_simulation(complete_model, prediction_horizon)

    with Simulation(network_path_name) as sim:

        if state_space_model["sim_type"] == SimType.EULER:
            euler_model.run_euler_model_simulation(time_step, complete_model, prediction_horizon, sim,
                                                   pump_ids, tank_ids, junction_ids)

        elif state_space_model["sim_type"] == SimType.PREISMANN:
            preismann_model.run_preismann_model_simulation(complete_model, prediction_horizon, sim)

    return network_df


def save_data(sim_df, simulation_type):
    """
    Saves DataFrames into folders named by date. Adds timestamp and simulation type into the name.
    Output is in terms of python pickles of the df.
    :param sim_df: The logs of the DataFrame from the simulation
    :param simulation_type: Type of the simulation
    :return: None
    """

    today = datetime.now()
    timestamp_day = today.strftime("%m_%d_%y")
    timestamp_hour = today.strftime("%H_%M_%S")
    path = f"data/mpc_data/{timestamp_day}"

    directory_contents = os.listdir("data/mpc_data/")
    if timestamp_day not in directory_contents:
        os.mkdir(path)

    # TODO: remove enum dot from name
    file_name = f"{path}/mpc_simulation_df_{timestamp_hour}_{simulation_type}.pkl"

    print(file_name)

    sim_df.to_pickle(file_name)


if __name__ == "__main__":
    print_welcome_msg()

    # create columns and pandas DataFrame to store data
    columns = ['timestamp', 'time_step', 'tank_volume', 'tank_depth', 'tank_overflow', 'tank_inflow', 'pump_flow']
    network_df = pd.DataFrame(columns=columns)

    # Select EPA SWMM network topology .inp
    network_name = "epa_networks/project_network/project_network.inp"

    # Configure the disturbance
    # If you do not wish to use any gains on the data,
    # set either rain_gain or poop_gain to 1
    # It also possible to select which disturbance you want to use
    # And later, I'll also add a functionality for the Gaussian gain into the model
    config = {
        "disturbance_data_name": "data/disturbance_data/hour_poop_rain.csv",
        "use_rain": True,
        "use_poop": True,
        "rain_gain": 13,
        "poop_gain": 10,
        # "use_random": False
    }

    disturb_manager = disturbance_reader.Disturbance(config)

    rows = disturb_manager.get_k_delta_disturbance(1, 10)

    # EPA SWMM engine step size [seconds]
    sim_step_size = 60

    # These have to be the names of the pumps and tanks from EPA SWMM
    pumps = ["FP1", "FP2"]
    tanks = ["T1", "T2"]
    junctions = ["N1", "N1", "N2", "N3", "N4", "N5"]

    # TODO: make this into a dictionary
    prediction_horizon = 10
    control_horizon = 10
    disturbance_magnitude = 5
    steps_between_plots = 3
    plot_mpc_steps = True

    # Make sure to select the right type of model you want to run the MPC on
    state_space_model = define_state_space_model(SimType.EULER, prediction_horizon,
                                                 disturbance_magnitude)

    complete_model = make_mpc_model(state_space_model, prediction_horizon, control_horizon)

    simulation_df = run_simulation(sim_step_size, pumps, tanks, junctions, network_df, network_name, complete_model,
                                   steps_between_plots, plot_mpc_steps)
    # simulation_df = network_df

    # save_data(simulation_df, prediction_model["sim_type"])

    print("Simulation is finished.")
