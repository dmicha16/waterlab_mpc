import casadi as ca
import numpy as np

from controller import mpco
from pyswmm import Simulation, Nodes, Links
from enum import Enum


class PumpSetting(Enum):
    CLOSED = 0
    # 2 is the max we allow physically
    OPEN = 2
    PUMP_2_OPEN = 1


def local_controller(tank, type, min_depth_tank):

    if type == "t1":
        if tank.depth >= min_depth_tank:
            return PumpSetting.OPEN.value
        else:
            return PumpSetting.CLOSED.value

    elif type == "t2":
        if tank.depth >= min_depth_tank:
            return PumpSetting.PUMP_2_OPEN.value
        else:
            return PumpSetting.CLOSED.value


def run_local_model_simulation(time_step, sim, pump_ids, tank_ids, junction_ids,
                               network_df, disturb_manager, num_sim_steps):

    pump1 = Links(sim)[pump_ids[0]]
    pump2 = Links(sim)[pump_ids[1]]
    tank1 = Nodes(sim)[tank_ids[0]]
    tank2 = Nodes(sim)[tank_ids[1]]

    disturb_pump = Links(sim)[pump_ids[2]]

    junction_n1 = Nodes(sim)[junction_ids[0]]
    junction_n2 = Nodes(sim)[junction_ids[1]]
    junction_n3 = Nodes(sim)[junction_ids[2]]
    junction_n4 = Nodes(sim)[junction_ids[3]]
    junction_n5 = Nodes(sim)[junction_ids[4]]

    network_elements = {
        "tank1_depth": tank1.depth,
        "tank1_volume": tank1.volume,
        "tank1_flooding": tank1.flooding,
        "tank1_inflow": tank1.total_inflow,
        "tank2_depth": tank2.depth,
        "tank2_volume": tank2.volume,
        "tank2_flooding": tank2.flooding,
        "tank2_inflow": tank2.total_inflow,
        "junction_n1_depth": junction_n1.depth,
        "junction_n2_depth": junction_n2.depth,
        "junction_n3_depth": junction_n3.depth,
        "junction_n4_depth": junction_n4.depth,
        "junction_n5_depth": junction_n5.depth,
        "pump1_flow": pump1.flow,
        "pump1_current_setting": pump1.current_setting,
        "pump1_target_setting": pump1.target_setting,
        "pump2_flow": pump2.flow,
        "pump2_current_setting": pump2.current_setting,
        "pump2_target_setting": pump2.target_setting,
        "disturbance": 0
    }

    # start the simulation with the pumps closed
    # https://pyswmm.readthedocs.io/en/stable/reference/nodes.html
    # use these functions to set how much the pump is open for
    pump1.target_setting = PumpSetting.CLOSED.value
    pump2.target_setting = PumpSetting.CLOSED.value

    for idx, step in enumerate(sim):

        current_disturb = disturb_manager.get_k_disturbance(idx, 1)

        disturb_pump.target_setting = current_disturb

        network_df = network_df.append(network_elements, ignore_index=True)

        if num_sim_steps <= 0:

            user_key_input = input("Press any key to step, or q to quit")
            try:
                num_sim_steps = int(user_key_input)
            except ValueError:
                pass

            if user_key_input == "q":
                break
        else:
            # print(num_sim_steps)
            num_sim_steps -= 1

        pump1.target_setting = local_controller(tank1, "t1", min_depth_tank=0.1)
        pump2.target_setting = local_controller(tank2, "t2", min_depth_tank=0.1)

        sim.step_advance(time_step)

        network_elements = {
            "tank1_depth": tank1.depth,
            "tank1_volume": tank1.volume,
            "tank1_flooding": tank1.flooding,
            "tank1_inflow": tank1.total_inflow,
            "tank2_depth": tank2.depth,
            "tank2_volume": tank2.volume,
            "tank2_flooding": tank2.flooding,
            "tank2_inflow": tank2.total_inflow,
            "junction_n1_depth": junction_n1.depth,
            "junction_n2_depth": junction_n2.depth,
            "junction_n3_depth": junction_n3.depth,
            "junction_n4_depth": junction_n4.depth,
            "junction_n5_depth": junction_n5.depth,
            "pump1_flow": pump1.flow,
            "pump1_current_setting": pump1.current_setting,
            "pump1_target_setting": pump1.target_setting,
            "pump2_flow": pump2.flow,
            "pump2_current_setting": pump2.current_setting,
            "pump2_target_setting": pump2.target_setting,
            "disturbance": float(current_disturb[0])
        }


    return network_df
