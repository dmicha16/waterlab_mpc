#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:12:58 2020

@author: davidm
"""


import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from pyswmm import Simulation, Nodes, Links
import networkcontrol as controller
import plotter as plotter

# %%

columns = ['tank_volume',
           'tank_depth',
           'tank_overflow',
           'tank_inflow',
           'pump_flow']

# %%
network_df = pd.DataFrame(columns=columns)

print(network_df)

# %%

count = 0

# step size [seconds]
Ts = 30

with Simulation('epa_networks/proposal_network/proposal_network.inp') as sim:
    
    sim.step_advance(Ts)

    pump1 = Links(sim)["P1"]
    tank1 = Nodes(sim)["Tanks1"]

    # Leave some volume in the tank to avoid dry-run of the pumps
    tank_mindepth = 0.1

    for idx, step in enumerate(sim):
        
        # call the controller here
        controller.on_off(pump1, tank1)

        network_df = network_df.append(pd.Series([tank1.volume, tank1.depth, tank1.flooding, tank1.total_inflow,
                                                  pump1.flow], index=network_df.columns), ignore_index=True)

        count += 1

        print(f"Progress {int(sim.percent_complete*100)}%", end="\r")

    t = np.linspace(0, count, count)
    plotter.plot_network(network_df, Ts, t, tank1.full_depth, count)

    plt.show()

    print('Count =', count)

