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

    Ap = ca.DM([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., -0.57690, 0.32141, 0.11616, -0.00806, -1.3343, 0.01643, 1.5745, -0.02942, -1.9671, 0.04957, -1.0914, 0.90771], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 1.3295, -0.25344, 0.63522, 0.04337, 1.3427, -0.01652, -1.5843, 0.02961, 1.9795, -0.04988, 1.0983, -0.91339], [0., 5.8510, -1.1153, 5.8806, -1.3277, -0.19762, 0.002434, 0.23319, -0.004359, -0.29133, 0.007338, -0.16164, 0.13443], [0., -0.92533, 0.17639, 0.52445, -0.06726, -0.6130, 0.05206, 1.6107, -0.03011, -2.0124, 0.05071, -1.1166, 0.92859], [0., -3.7105, 0.70732, 2.1033, -0.26973, 6.1943, -1.3306, -0.39037, 0.007293, 0.48773, -0.01229, 0.27060, -0.22506], [0., 0.66951, -0.12762, -0.37947, 0.04867, 1.6334, -0.07625, -0.8962, 0.06604, 2.0594, -0.05189, 1.1426, -0.95026], [0., 2.3410, -0.44621, -1.3269, 0.17023, 5.7112, -0.26658, 6.3626, -1.3358, -0.62377, 0.01572, -0.34608, 0.28783], [0., -0.50862, 0.096954, 0.28827, -0.03697, -1.2409, 0.05792, 1.9303, -0.09067, -1.3642, 0.08856, -1.1742, 0.97652], [0., -1.4643, 0.27913, 0.82986, -0.10640, -3.5725, 0.16679, 5.5571, -0.26105, 6.5851, -1.3447, 0.40035, -0.33296], [0., 0.44672, -0.085151, -0.25320, 0.032479, 1.0899, -0.050867, -1.6953, 0.079624, 2.6426, -0.12460, 2.1912, -1.5796], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    Bp = ca.DM([[-11.886, 0., -11.886, 0.], [0.21376, 1.1940, 0., 268.53], [1.0000, 0., 0., 0.], [-0.14457, -1.2015, 0., -270.20], [-0.63619, 0.17683, 0., 39.769], [0.10062, 1.2215, 0., 274.71], [0.40347, -0.29604, 0., -66.578], [-0.072797, -1.2500, 0., -281.12], [-0.25455, 0.37861, 0., 85.147], [0.055305, 1.2845, 0., 288.88], [0.15921, -0.43798, 0., -98.503], [-0.048574, -2.4988, 0., -293.05], [0., 1.0000, 0., 0.]])

    Bp_d = ca.DM([11.88589540, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    # ca.DM([0., -0.028882, 0., 0., 0., -0.10716, 0.13604])
    operating_point = ca.DM([0., 1.8571, 0., -1.7729, -1.9879, 1.7396, -1.1978, -1.7393, -1.2053, 1.7608, -1.7156, -2.8334, 0.]) * 1

    # [0.152070692682436]

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
    # lower_bounds_states = ca.DM([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # upper_bounds_input = ca.DM([2, 2, ca.inf, ca.inf])
    # upper_bounds_slew_rate = ca.DM([1, 1, ca.inf, ca.inf])
    upper_bounds_states = ca.DM([3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 2, 3]) * 5

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
