import casadi as ca
import numpy as np

from controller import mpco
from pyswmm import Simulation, Nodes, Links
from enum import Enum
import pandas as pd

from util_scripts import disturbance_reader


class PumpSetting(Enum):
    CLOSED = 0
    OPEN = 1


def set_preismann_initial_conditions():
    """
    Set initial conditions for the Preismann model
    :return: x0 and u0
    """

    x0 = ca.DM.zeros(13, 1)
    u0 = ca.DM.zeros(4, 1)

    return [x0, u0]


def set_preismann_weight_matrices():
    """
    Set the Q and R weight matrices for the Preismann model
    :return: Q and R matrices
    """

    Q = ca.DM(np.identity(13)) * 1
    Q[0, 0] = 10
    Q[12, 12] = 10

    R = ca.DM([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    S = ca.DM([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 10000, 0], [0, 0, 0, 10000]])

    return [Q, R, S]


def set_preismann_ref(prediction_horizon, num_states):
    """
    Create an arbitrary reference for the Preismann model
    :param prediction_horizon: Length of the prediction horizon.
    :param num_states: The number of states in the model
    :return: Reference corresponding to the Preismann model
    """

    ref = ca.DM.ones(prediction_horizon * num_states, 1)
    constant_ref = ca.DM([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    for state in range(num_states):
        ref[state::num_states] = ref[state::num_states] * constant_ref[state]

    return ref


def make_preismann_model(simulation_type, pred_horizon):
    """
    Construct the Preismann model based state space model, using the prediction horizon and the
    disturbance magnitude
    :param simulation_type: This is to denote in the state_space_mode|l which type of simulation it is.
    :param pred_horizon: Length of the prediction horizon
    :return: Entire state space model of the Preismann model with both the initial conditions and with the
    weight matrices.
    """

    Ap = ca.DM([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., -6.3263, -0.33994, -13.824, 0.61062, -27.063, 0.83032, -35.031, 1.1406, -49.841, 1.5559, -20.440,
                 9.4870], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [0., 6.2280, 0.46601, 13.986, -0.57254, 27.944, -0.85729, 36.170, -1.1778, 51.461, -1.6065, 21.105,
                 -9.7957],
                [0., 2.5976, 0.19437, 1.9828, -0.58256, -2.0827, 0.063896, -2.6958, 0.087794, -3.8355, 0.11972, -1.5730,
                 0.73008],
                [0., -6.1299, -0.45865, -14.514, 0.63883, -29.382, 0.95057, -37.659, 1.2263, -53.579, 1.6725, -21.974,
                 10.199],
                [-0., 2.3652, 0.17698, 5.6003, -0.24647, 3.6851, -0.64916, 0.82651, -0.026913, 1.1759, -0.036708,
                 0.48227, -0.22383],
                [0., 6.6036, 0.49410, 15.636, -0.68822, 29.582, -0.94153, 38.694, -1.2113, 55.460, -1.7315, 22.745,
                 -10.557],
                [0., 1.2452, 0.093167, 2.9485, -0.12975, 5.5782, -0.17759, 0.74828, -0.55471, -3.2697, 0.10206, -1.3410,
                 0.62237],
                [0., -6.6739, -0.49935, -15.802, 0.69543, -29.898, 0.95163, -41.048, 1.3016, -57.930, 1.8572, -23.641,
                 10.972],
                [0., 1.4121, 0.10566, 3.3436, -0.14715, 6.3260, -0.20135, 8.6853, -0.27540, 4.9410, -0.68870, 0.77691,
                 -0.36058],
                [0., 7.2708, 0.54402, 17.216, -0.75763, 32.572, -1.0369, 44.717, -1.4180, 61.021, -1.9402, 24.969,
                 -11.714], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    Bp = ca.DM([[0., -11.886, 0., 0.], [0., -1.3349, 0., 17.329], [0., 1.0000, 0., 0.], [0., 1.4942, 0., -17.893],
                [0., 0.62320, 0., 1.3336], [0., -1.4706, 0., 18.630], [-0., 0.56739, -0., -0.40886],
                [0., 1.5842, 0., -19.284], [0., 0.29874, 0., 1.1368], [0., -1.6011, 0., 20.042],
                [0., 0.33878, 0., -0.65866], [0., 1.7444, 0., -21.514], [0., 0., 0., 1.0000]])

    Bp_d = ca.DM([11.88589540, 0., 0., 0., 0., 0., -0., 0., 0., 0., 0., 0., 0.])

    # ca.DM([0., -0.028882, 0., 0., 0., -0.10716, 0.13604])
    operating_point = ca.DM(
        [0., 0., 0., -0.51186, 0., -0.51186, 0., -0.51186, 0., -0.51186, -0.039620, -0.93444, 0.]) * 0

    # Un-constraint
    lower_bounds_input = None
    lower_bounds_slew_rate = None
    lower_bounds_states = None
    upper_bounds_input = None
    upper_bounds_slew_rate = None
    upper_bounds_states = None

    # Actual constraints
    lower_bounds_input = ca.DM([0, 0, 0, 0])
    # lower_bounds_slew_rate = ca.DM([-ca.inf, -ca.inf, -ca.inf, -ca.inf])
    lower_bounds_states = ca.DM([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    upper_bounds_input = ca.DM([2, 2, ca.inf, ca.inf])
    # upper_bounds_slew_rate = ca.DM([1, 1, ca.inf, ca.inf])
    upper_bounds_states = ca.DM([3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 3]) * 10

    # size1 and size2 represent the num of rows and columns in the Casadi lib, respectively
    num_states = Ap.size1()
    num_inputs = Bp.size2()

    # Initial conditions here are needed for the right dimensions in the MPC
    [x0, u0] = set_preismann_initial_conditions()
    [Q, R, S] = set_preismann_weight_matrices()

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

    print("Preismann model is constructed.")

    return initial_state_space_model


def make_preismann_mpc_model(state_space_model, prediction_horizon, control_horizon):
    """
    Make the MPC model for the Preismann state space model. This function exposes the interface
    to the MPC class instance, so it is easy to see what is being passed as the arguments from the state
    space model. Furthermore, the control and prediction horizon is also added.
    :param state_space_model: Preismann model state space model.
    :param prediction_horizon: Length of the prediction horizon.
    :param control_horizon: Length of the control horizon
    :return: Augmented state space model dictionary with the state space model as a new entry.
    """

    num_states = state_space_model["num_states"]

    ref = set_preismann_ref(prediction_horizon, num_states)

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


def run_preismann_model_simulation(time_step, complete_model, prediction_horizon, sim, pump_ids, tank_ids, junction_ids,
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
    pump1.target_setting = int(0)
    pump2.target_setting = int(0)

    # x initial conditions
    states = [
        tank1.initial_depth,
        junction_n1.initial_depth,
        junction_n1.lateral_inflow,
        junction_n2.initial_depth,
        junction_n2.lateral_inflow,
        junction_n3.initial_depth,
        junction_n3.lateral_inflow,
        junction_n4.initial_depth,
        junction_n3.lateral_inflow,
        junction_n5.initial_depth,
        junction_n3.lateral_inflow,
        tank2.initial_depth,
        tank2.lateral_inflow
    ]

    # u_prev, initial control input
    control_input = ca.DM.zeros(complete_model["num_inputs"], 1)

    # zeros for now
    # TODO: make sure to add a proper reference
    ref = set_preismann_ref(prediction_horizon, complete_model["num_states"])

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

        states = [
            tank1.depth,
            junction_n1.depth,
            junction_n1.total_inflow,
            junction_n2.depth,
            junction_n2.total_inflow,
            junction_n3.depth,
            junction_n3.total_inflow,
            junction_n4.depth,
            junction_n3.total_inflow,
            junction_n5.depth,
            junction_n3.total_inflow,
            tank2.depth,
            tank2.total_inflow
        ]

    return network_df
