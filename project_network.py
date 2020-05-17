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

from datetime import datetime
from pyswmm import Simulation, Nodes, Links
import networkcontrol as controller
from controller import mpc, mpco
# import plotter as plotter


def print_welcome_msg():
    # TODO: add ascii art ;)

    print("A python project of a Model Predictive Controller (MPC)"
          " for Urban Drainage Networks (UDNs) to mitigate Combined Sewer Overflows (CSOs)")


def define_prediction_model():

    Ap = ca.DM([[1, 0, 0], [0, 0.5, 0], [0, 0, 0.7]])
    Bp = ca.DM([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    Bp_d = ca.DM([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 1]])

    states = Ap.size1()
    inputs = Bp.size2()

    # TODO: fix the names here
    prediction_model = {"system_model": Ap, "b_matrix": Bp, "bp_d": Bp_d, "states": states, "inputs": inputs}

    return prediction_model


def define_real_model(prediction_model, prediction_horizon, control_horizon, disturbance_magnitude):

    Ap = prediction_model["system_model"]
    Bp = prediction_model["b_matrix"]
    Bp_d = prediction_model["bp_d"]

    states = prediction_model["states"]
    inputs = prediction_model["inputs"]

    rand_var = 0.00
    const_var = 1.00
    rand_A = np.random.rand(Ap.size1(), Ap.size2())

    A = Ap * const_var + (np.random.rand(Ap.size1(), Ap.size2()) - 0.5) * rand_var
    B = Bp * const_var + (np.random.rand(Bp.size1(), Bp.size2()) - 0.5) * rand_var
    B_d = Bp_d * const_var + (np.random.rand(Bp_d.size1(), Bp_d.size2()) - 0.5) * rand_var

    dist = (np.random.rand(prediction_horizon * inputs) - 0.5) * disturbance_magnitude

    Q = ca.DM([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    Qb = mpc.blockdiag(Q, prediction_horizon)
    R = ca.DM([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
    Rb = mpc.blockdiag(R, control_horizon)

    x0 = ca.DM([[10], [11], [12]])
    u0 = ca.DM([-1, -3, -6])

    ref = ca.DM.ones(prediction_horizon * states, 1)
    for state in range(states):
        ref[state::states] = ref[state::states] + state - 2

    return mpco.MpcObj(Ap, Bp, control_horizon, prediction_horizon, Q, R, x0, u0, ref, Bp_d, dist)


def run_simulation(time_step, pumps, tanks, network_df, network_name, mpc_model):

    with Simulation(network_name) as sim:

        links = Links(sim)
        nodes = Nodes(sim)

        for pump in pumps:
            if pump not in links:
                print("This pump doesnt exist")

        for tank in tanks:
            if tanks not in nodes:
                print("This tank doesnt exist")

        sim.step_advance(time_step)

        for idx, step in enumerate(sim):

            # step
            # get most recent log
            # mylist [epa_simulation_stuff, mpc]
            network_df = network_df.append(pd.Series([tank1.volume, tank1.depth, tank1.flooding, tank1.total_inflow,
                                                      pump1.flow], index=network_df.columns), ignore_index=True)

            print(f"Progress {int(sim.percent_complete * 100)}%", end="\r")


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

    mpc_model = define_real_model(prediction_model, prediction_horizon, control_horizon, disturbance_magnitude)

    simulation_df = run_simulation(time_step, pumps, tanks, network_df, network_name, mpc_model)
    # simulation_df = network_df

    save_data(simulation_df)







