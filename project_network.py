#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 20:12:58 2020

@author: davidm
"""

import matplotlib.pyplot as plt

import pandas as pd
from datetime import datetime

import os

from pyswmm import Simulation, Nodes, Links
import networkcontrol as controller
import plotter as plotter



def print_welcome_msg():
    # TODO: add ascii art ;)

    print("A python project of a Model Predictive Controller (MPC)"
          " for Urban Drainage Networks (UDNs) to mitigate Combined Sewer Overflows (CSOs)")


def run_simulation(time_step, pumps, tanks, network_df):

    with Simulation('epa_networks/project_network/project_network.inp') as sim:

        sim.step_advance(time_step)

        for idx, step in enumerate(sim):
            pass


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
    columns = ['tank_volume', 'tank_depth', 'tank_overflow', 'tank_inflow', 'pump_flow']
    network_df = pd.DataFrame(columns=columns)

    # step size [seconds]
    time_step = 30

    pumps = []
    tanks = []

    # simulation_df = run_simulation(time_step, pumps, tanks, network_df)
    simulation_df = network_df

    save_data(simulation_df)







