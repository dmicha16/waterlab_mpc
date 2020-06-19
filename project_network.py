#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 20:12:58 2020

@author: davidm
"""

import os
import json

import pandas as pd

from datetime import datetime
from pyswmm import Simulation
from util_scripts import disturbance_reader
import tikz_plot_generator

from enum import Enum

from state_space_models import euler_model, \
    custom_model, preismann_model, local_model


class SimType(Enum):
    EULER = 1
    PREISMANN = 2
    CUSTOM_MODEL = 3
    LOCAL_MODEL = 4


def print_welcome_msg():
    """
    Prints a welcome msg and general status of the project.
    :return: None
    """
    # TODO: add ascii art ;)
    # TODO: add perhaps version of the code base?

    print("A python project of a Model Predictive Controller (MPC)"
          " for Urban Drainage Networks (UDNs) to mitigate Combined Sewer Overflows (CSOs)")


def define_state_space_model(simulation_type, pred_horizon):
    """
    Interface to the specific model declarations. The appropriate model construction is called
    specified by the simulation type passed to this function.
    :param simulation_type: Enum to switch between model types.
    :param pred_horizon: Length of the prediction_horizon.
    :return: The constructed initial state space model.
    """

    if simulation_type == SimType.EULER:
        initial_state_space_model = euler_model.make_euler_model(simulation_type, pred_horizon)

    elif simulation_type == SimType.PREISMANN:
        initial_state_space_model = preismann_model.make_preismann_model(simulation_type, pred_horizon)

    elif simulation_type == SimType.CUSTOM_MODEL:
        initial_state_space_model = custom_model.make_custom_model_model(simulation_type, pred_horizon)

    elif simulation_type == SimType.LOCAL_MODEL:
        initial_state_space_model = {
            "sim_type": SimType.LOCAL_MODEL
        }


    else:
        print("Default, going with generic model.")
        initial_state_space_model = custom_model.make_custom_model_model(simulation_type, pred_horizon)

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
    elif simulation_type == SimType.LOCAL_MODEL:
        aug_state_space_model = ss_model

    else:
        aug_state_space_model = preismann_model.make_preismann_mpc_model(ss_model, pred_horizon,
                                                                         ctrl_horizon)

    return aug_state_space_model


def run_simulation(simul_config, data_df, complete_sys_model, disturb_manager):

    if simul_config["sim_type"] == SimType.CUSTOM_MODEL:
        custom_model.run_custom_model_simulation(complete_sys_model, simul_config["prediction_horizon"])

    with Simulation(simul_config["network_name"]) as sim:

        if simul_config["sim_type"] == SimType.EULER:
            data_df = euler_model.run_euler_model_simulation(simul_config["sim_step_size"],
                                                             complete_sys_model,
                                                             simul_config["prediction_horizon"],
                                                             sim,
                                                             simul_config["pumps"],
                                                             simul_config["tanks"],
                                                             simul_config["junctions"],
                                                             data_df,
                                                             disturb_manager,
                                                             simul_config["num_plot_steps"])

        elif simul_config["sim_type"] == SimType.PREISMANN:
            data_df = preismann_model.run_preismann_model_simulation(simul_config["sim_step_size"],
                                                                     complete_sys_model,
                                                                     simul_config["prediction_horizon"],
                                                                     sim,
                                                                     simul_config["pumps"],
                                                                     simul_config["tanks"],
                                                                     simul_config["junctions"],
                                                                     data_df,
                                                                     disturb_manager,
                                                                     simul_config["num_plot_steps"])

        elif simul_config["sim_type"] == SimType.LOCAL_MODEL:
            data_df = local_model.run_local_model_simulation(simul_config["sim_step_size"],
                                                             sim,
                                                             simul_config["pumps"],
                                                             simul_config["tanks"],
                                                             simul_config["junctions"],
                                                             data_df,
                                                             disturb_manager,
                                                             simul_config["num_plot_steps"])

        else:
            # There is no other type of simulation, you should probably make sure you selected the correct one.
            print("No simulation is selected.")
            pass

    return data_df


def save_data(sim_df, simulation_config, disturbance_config):
    """
    Saves DataFrames into folders named by date. Adds timestamp and simulation type into the name.
    Output is in terms of python pickles of the df.
    :param sim_df: The logs of the DataFrame from the simulation
    :param simulation_config:
    :param: disturb_config:
    :return: None
    """

    simulation_type = simulation_config["sim_type"].name
    simulation_config["sim_type"] = simulation_config["sim_type"].name

    today = datetime.now()
    timestamp_day = today.strftime("%m_%d_%y")
    timestamp_hour = today.strftime("%H_%M_%S")
    path = f"data/mpc_data/{timestamp_day}"

    directory_contents = os.listdir("data/mpc_data/")
    if timestamp_day not in directory_contents:
        os.mkdir(path)

    # TODO: remove enum dot from name
    file_name = f"{path}/mpc_simulation_df_{timestamp_hour}_{simulation_type}.pkl"
    json_file_name = f"{path}/mpc_simulation_df_{timestamp_hour}_{simulation_type}.json"

    # It is possible to just simply add more info the json with this one dictionary below, by updating the entries
    combined_dict = {
        "date": timestamp_day,
        "hour": timestamp_hour
    }
    combined_dict.update(simulation_config)
    combined_dict.update(disturbance_config)

    sim_df.to_pickle(file_name)

    with open(json_file_name, 'w') as json_file:
        json.dump(combined_dict, json_file, indent=4)


if __name__ == "__main__":
    print_welcome_msg()

    # create columns and pandas DataFrame to store data
    columns = [
        'timestamp',
        'time_step',
        'tank1_depth',
        'tank1_volume',
        'tank1_flooding',
        'tank1_inflow',
        'tank2_depth',
        'tank2_volume',
        'tank2_flooding',
        'tank2_inflow',
        'junction_n1_depth',
        'junction_n2_depth',
        'junction_n3_depth',
        'junction_n4_depth',
        'junction_n5_depth',
        'pump1_flow',
        'pump1_current_setting',
        'pump1_target_setting',
        'pump2_flow'
        'pump2_current_setting'
        'pump2_target_setting'
        'disturbance'
    ]
    network_df = pd.DataFrame(columns=columns)

    sim_config = {

        # Select EPA SWMM network topology .inp
        "network_name": "epa_networks/project_network/project_network.inp",

        # EPA SWMM engine step size [seconds]
        "sim_type": SimType.LOCAL_MODEL,
        "sim_step_size": 12.65,

        # Number of steps before the plotter plots anything
        "num_plot_steps": 150,

        # These have to be the names of the pumps and tanks from EPA SWMM
        "pumps": ["FP1", "FP2", "FP3"],
        "tanks": ["T1", "T2"],
        "junctions": ["N1", "N1", "N2", "N3", "N4", "N5"],

        # MPC related configuration
        "prediction_horizon": 2,
        "control_horizon": 2,
        "steps_between_plots": 3,
        "plot_mpc_steps": True
    }

    # Configure the disturbance
    # If you do not wish to use any gains on the data,
    # set either rain_gain or poop_gain to 1.
    # It also possible to select which disturbance you want to use
    disturb_config = {
        "disturbance_data_name": "data/disturbance_data/hour_poop_rain.csv",
        "use_rain": True,
        "use_poop": True,
        "rain_gain": 5.6,
        "poop_gain": 1.12,
        # "use_random": False
    }

    disturb_manager = disturbance_reader.Disturbance(disturb_config)

    # Make sure to select the right type of model you want to run the MPC on
    state_space_model = define_state_space_model(sim_config["sim_type"], sim_config["prediction_horizon"], )

    complete_model = make_mpc_model(state_space_model, sim_config["prediction_horizon"], sim_config["control_horizon"])

    simulation_df = run_simulation(sim_config, network_df, complete_model, disturb_manager)

    save_data(simulation_df, sim_config, disturb_config)

    tikz_plot_generator.plot_df(simulation_df)

    print("Simulation is finished.")

    # TODO: replace the euler model with the updated values
    # TODO: perhaps add commits ;)
