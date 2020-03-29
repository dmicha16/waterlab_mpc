import pandas as pd
import numpy as np
from pyswmm import Simulation, Nodes, Subcatchments, Links
import sewerPlot as sewerPlot
import sewerControl as sewerControl

# This is called a magic and you can use to do magical things
# in this case allowing matplotlib to show in jupyter lab
#get_ipython().run_line_magic('matplotlib', 'inline')

# Preallocate
count = 0
tank_volume = []
tank_depth = []
tank_overflow = []
tank_inflow = []
pump_flow = []
tank_maxdepth = []

with Simulation('test_network.inp') as simulator:
    
    # Create objects from string-search
    pumps = [Links(simulator)["Pump1"], Links(simulator)["Pump2"]]
    tanks = [Nodes(simulator)["Tank1"], Nodes(simulator)["Tank2"]]

    # Initial conditions
    pumps[0].initial_flow, pumps[1].initial_flow = [0, 0]
    tanks[0].initial_depth, tanks[1].initial_depth = [0.5, 0.7]
    
    tank_mindepth = [0.1,0.1]   # Leave some volume in the tank to avoid dry-run of the pumps                              
    tank_maxdepth = [tanks[0].full_depth, tanks[1].full_depth]
    
    # Sampling time [s] 
    Ts = 300
    simulator.step_advance(Ts)
    
    for step in enumerate(simulator):
        
        # On/Off control: 
        sewerControl.on_off(tanks,pumps,tank_maxdepth,tank_mindepth)
        # MPC: 
        # coming soon 

        count += 1
     
        # Read variables
        tank_volume.append([tanks[0].volume, tanks[1].volume])
        tank_depth.append([tanks[0].depth, tanks[1].depth])
        tank_overflow.append([tanks[0].flooding, tanks[1].flooding])                
        pump_flow.append([pumps[0].flow, pumps[1].flow])  
        tank_inflow.append([tanks[0].total_inflow, tanks[1].total_inflow])
        
        print(int(simulator.percent_complete*100), '%')
        
    print('Count =', count)
    
    Of1 = np.sum(np.array(tank_overflow)[:,0])
    Of2 = np.sum(np.array(tank_overflow)[:,1])
    
    print('Overflow at station 1:', Of1, '[m3]')
    print('Overflow at station 2:', Of2, '[m3]')

####################################################### Pre-processing #####################################################
    
    t = np.linspace(0, count, count)
    sewerPlot.sPlotter(tank_inflow,tank_overflow,tank_volume,tank_depth,tank_maxdepth,pump_flow,count,t,Ts)

