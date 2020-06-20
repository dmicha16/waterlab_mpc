import casadi as ca
import numpy as np

from controller import mpco
from pyswmm import Simulation, Nodes, Links
from enum import Enum

import pandas as pd


class PumpSetting(Enum):
    CLOSED = 0
    OPEN = 1


def set_euler_initial_conditions():
    """
    Set initial conditions for the Euler model. In this case it is only to
    determine the correct dimensions for the MPC
    :return: x0 and u0
    """

    x0 = ca.DM.zeros(7, 1)
    u0 = ca.DM.zeros(4, 1)

    return [x0, u0]


def set_euler_weight_matrices():
    """
    Set the Q and R weight matrices for the Euler model
    :return: Q and R matrices
    """

    # initially to ones to run the code
    # TODO: add proper Q and R matrices @Casper
    Q = ca.DM(np.identity(7)) * 1

    Q[0, 0] = 10
    Q[6, 6] = 10
    R = ca.DM(np.zeros())
    R = ca.DM([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, 0]])

    S = ca.DM([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 10000, 0],
               [0, 0, 0, 10000]])

    L = ca.DM([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 10000, 0],
               [0, 0, 0, 10000]])

    return [Q, R, S, L]


def set_euler_ref(prediction_horizon, num_states):
    """
        Create an linear arbitrary reference for the Euler model
        :param prediction_horizon: Length of the prediction horizon.
        :param num_states: The number of states in the model
        :return: Reference corresponding to the Euler model
        """

    ref = ca.DM.ones(prediction_horizon * num_states, 1)
    constant_ref = ca.DM([0, 0, 0, 0, 0, 0, 0])

    for state in range(num_states):
        ref[state::num_states] = ref[state::num_states] * constant_ref[state]
    return ref


def make_euler_model(simulation_type, pred_horizon):
    """
    Construct the Euler model based state space model, using the prediction horizon and the
    disturbance magnitude
    :param simulation_type: This is to denote in the state_space_model which type of simulation it is.
    :param pred_horizon: Length of the prediction horizon
    :return: Entire state space model of the Euler model with both the initial conditions and with the
    weight matrices.
    """

<<<<<<< Updated upstream
    Ap = ca.DM([[1., 0., 0., 0., 0., 0., 0.], [0., 0.84717, 0.0013996, 0., 0., 0., 0.], [0., 0.15283, 0.84577, 0.0013996, 0., 0., 0.], [0., 0., 0.15283, 0.84577, 0.0013996, 0., 0.], [0., 0., 0., 0.15283, 0.84577, 0.0013996, 0.], [0., 0., 0., 0., 0.15283, 0.80758, 0.039586], [0., 0., 0., 0., 0., 0.19102, 0.96041]])

    Bp = ca.DM([[-0.08413333333, 0, -0.08413333333, 0], [0.03155000000, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, -0.06310000000, 0, -0.06310000000]])
=======
    Ap = ca.DM([[1., 0., 0., 0., 0., 0., 0.],
                [0., 0.86386, 0.092822, 0., 0., 0., 0.],
                [0., 0.13614, 0.77104, 0.092822, 0., 0., 0.],
                [0., 0., 0.13614, 0.77104, 0.092822, 0., 0.],
                [0., 0., 0., 0.13614, 0.77104, 0.092822, 0.],
                [0., 0., 0., 0., 0.13614, 0.60133, 0.26253],
                [0., 0., 0., 0., 0., 0.30585, 0.73747]])

    Bp = ca.DM([[-1 / 15, 0, -1 / 15, 0],
                [1 / 16, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, -1 / 20, 0, -1 / 20]])
    Bp = ca.horzcat(Bp, ca.DM(np.identity(Bp.size1())))
>>>>>>> Stashed changes

    Bp_d = ca.DM([0.08413333333, 0, 0, 0, 0, 0, 0])

    # ca.DM([0., -0.028882, 0., 0., 0., -0.10716, 0.13604])
    # We ignore the operating point until we can soften the constraints
    operating_point = ca.DM([0., 0.028683, 0., 0., 0., -0.039586, 0.010903]) * 0
    # Un-constraint
    lower_bounds_input = None
    lower_bounds_slew_rate = None
    lower_bounds_states = None
    upper_bounds_input = None
    upper_bounds_slew_rate = None
    upper_bounds_states = None

    # Actual constraints
    lower_bounds_input = ca.DM([0, 0, 0, 0])
    lower_bounds_slew_rate = ca.DM([-ca.inf, -ca.inf, -ca.inf, -ca.inf])
    lower_bounds_states = ca.DM([0, 0, 0, 0, 0, 0, 0])
    upper_bounds_input = ca.DM([2, 2, ca.inf, ca.inf])
    upper_bounds_slew_rate = ca.DM([1, 1, ca.inf, ca.inf])
    upper_bounds_states = ca.DM([3, 1, 1, 1, 1, 1, 2])

    # size1 and size2 represent the num of rows and columns in the Casadi lib, respectively
    num_states = Ap.size1()
    num_inputs = Bp.size2()

    # Initial conditions here are needed for the right dimensions in the MPC
    [x0, u0] = set_euler_initial_conditions()
    [Q, R, S] = set_euler_weight_matrices()

    initial_state_space_model = {
        "system_model": Ap,
        "b_matrix": Bp,
        "b_disturbance": Bp_d,
        "Q": Q,
        "R": R,
        "S": S,
        "x0": x0,
        "u0": u0,
        "operating_point": operating_point,
        "num_states": num_states,
        "num_inputs": num_inputs,
        "sim_type": simulation_type,
        "lower_bounds_input": lower_bounds_input,
        "lower_bounds_slew_rate": lower_bounds_slew_rate,
        "lower_bounds_states": lower_bounds_states,
        "upper_bounds_input": upper_bounds_input,
        "upper_bounds_slew_rate": upper_bounds_slew_rate,
        "upper_bounds_states": upper_bounds_states
    }

    print("Euler model is constructed.")

    return initial_state_space_model


def make_euler_mpc_model(state_space_model, prediction_horizon, control_horizon):
    """
    Make the MPC model for the Euler state space model. This function exposes the interface
    to the MPC class instance, so it is easy to see what is being passed as the arguments from the state
    space model. Furthermore, the control and prediction horizon is also added.
    :param state_space_model: Euler model state space model.
    :param prediction_horizon: Length of the prediction horizon.
    :param control_horizon: Length of the control horizon
    :return: Augmented state space model dictionary with the state space model as a new entry.
    """

    num_states = state_space_model["num_states"]

    ref = set_euler_ref(prediction_horizon, num_states)

    mpc_model = mpco.MpcObj(state_space_model["system_model"],
                            state_space_model["b_matrix"],
                            control_horizon,
                            prediction_horizon,
                            state_space_model["Q"],
                            state_space_model["R"],
                            input_cost=state_space_model["S"],
                            initial_control_signal=state_space_model["u0"],
                            ref=ref,
                            operating_point=state_space_model["operating_point"],
                            input_matrix_d=state_space_model["b_disturbance"],
                            lower_bounds_input=state_space_model["lower_bounds_input"],
                            lower_bounds_slew_rate=state_space_model["lower_bounds_slew_rate"],
                            lower_bounds_states=state_space_model["lower_bounds_states"],
                            upper_bounds_input=state_space_model["upper_bounds_input"],
                            upper_bounds_slew_rate=state_space_model["upper_bounds_slew_rate"],
                            upper_bounds_states=state_space_model["upper_bounds_states"]
                            )

    state_space_model["mpc_model"] = mpc_model

    return state_space_model


def run_euler_model_simulation(time_step, complete_model, prediction_horizon, sim, pump_ids, tank_ids, junction_ids,
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
        "pump2_target_setting": pump2.target_setting
    }

    mpc_model = complete_model["mpc_model"]

    # start the simulation with the pumps closed
    # https://pyswmm.readthedocs.io/en/stable/reference/nodes.html
    # use these functions to set how much the pump is open for
    pump1.target_setting = PumpSetting.CLOSED.value
    pump2.target_setting = PumpSetting.CLOSED.value

    # x initial conditions
    states = [
        tank1.initial_depth,
        junction_n1.initial_depth,
        junction_n2.initial_depth,
        junction_n3.initial_depth,
        junction_n4.initial_depth,
        junction_n5.initial_depth,
        tank2.initial_depth
    ]

    # u_prev, initial control input
    control_input = ca.DM.zeros(complete_model["num_inputs"], 1)

    # zeros for now
    # TODO: make sure to add a proper reference
    ref = set_euler_ref(prediction_horizon, complete_model["num_states"])

    # To make the simulation precise,
    # make sure that the flow metrics are in Cubic Meters per Second [CMS]
    for idx, step in enumerate(sim):

        future_delta_disturb = disturb_manager.get_k_delta_disturbance(idx, prediction_horizon)
        prev_disturb = disturb_manager.get_k_disturbance(idx - 1, 1)
        current_disturb = disturb_manager.get_k_disturbance(idx, 1)

        disturb_pump.target_setting = current_disturb

        network_df = network_df.append(network_elements, ignore_index=True)

        if num_sim_steps <= 0:

            mpc_model.plot_progress(options={'drawU': 'U'})
            user_key_input = input("Press any key to step, or q to quit")
            try:
                num_sim_steps = int(user_key_input)
            except ValueError:
                pass

            if user_key_input == "q":
                break
        else:
            print(num_sim_steps)
            num_sim_steps -= 1

        mpc_model.step(states, control_input, disturbance=future_delta_disturb, prev_disturbance=prev_disturb)

        control_input = control_input + mpc_model.get_next_control_input_change()
        pump1.target_setting = control_input[0]
        pump2.target_setting = control_input[1]

        sim.step_advance(time_step)

        # tank1.depth, hp1.depth, hp2.depth ... hp5.depth, tank2.depth
        # this here is a python stl list, you cannot subtract like this safely

        states = [
            tank1.depth,
            junction_n1.depth,
            junction_n2.depth,
            junction_n3.depth,
            junction_n4.depth,
            junction_n5.depth,
            tank2.depth
        ]

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
