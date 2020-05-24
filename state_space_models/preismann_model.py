import casadi as ca
import numpy as np

from controller import mpco
from pyswmm import Simulation, Nodes, Links
from enum import Enum
import pandas as pd


def set_preismann_initial_conditions():
    """
    Set initial conditions for the Preismann model
    :return: x0 and u0
    """

    # TODO: ID how many states there are
    x0 = ca.DM.zeros(7, 1)
    u0 = ca.DM.zeros(4, 1)

    return [x0, u0]


def set_preismann_weight_matrices():
    """
    Set the Q and R weight matrices for the Preismann model
    :return: Q and R matrices
    """

    # TODO: add proper Q and R matrices @Casper
    Q = ca.DM(np.identity(7)) * 0
    # TODO: take  look at the error please, you might need to use some casadi specific operations, because
    #  now I think it's trying to default to python only syntax with Q[1,1]
    Q[0, 0] = 1
    Q[6, 6] = 1
    R = ca.DM([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 10000, 0], [0, 0, 0, 10000]])

    return [Q, R]


def set_preismann_ref(prediction_horizon, num_states):
    """
        Create an arbitrary reference for the Preismann model
        :param prediction_horizon: Length of the prediction horizon.
        :param num_states: The number of states in the model
        :return: Reference corresponding to the Preismann model
        """

    ref = ca.DM.ones(prediction_horizon * num_states, 1)
    constant_ref = ca.DM([0, 0, 0, 0, 0, 0, 0])

    for state in range(num_states):
        ref[state::num_states] = ref[state::num_states] * constant_ref[state]

    return ref


def make_preismann_model(simulation_type, pred_horizon, disturb_magnitude):
    """
    Construct the Preismann model based state space model, using the prediction horizon and the
    disturbance magnitude
    :param simulation_type: This is to denote in the state_space_mode|l which type of simulation it is.
    :param pred_horizon: Length of the prediction horizon
    :param disturb_magnitude: Magnitude of the disturbances
    :return: Entire state space model of the Preismann model with both the initial conditions and with the
    weight matrices.
    """

    # TODO: replace with the actualy preismann model here:
    Ap = ca.DM([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0.39870, 0.76314, 0.59389, 0.27052, -0.22723, -0.099109, 0.082650, 0.036507, -0.028479, -0.014028, 0.019530, 0.0038512], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 0.39550, -0.38686, 0.40362, -0.03194, 0.36180, 0.15782, -0.13160, -0.058130, 0.045346, 0.022337, -0.031096, -0.006132], [0., 0.63317, -0.61933, 0.00759, -1.2726, -0.41409, -0.18062, 0.15060, 0.066524, -0.051898, -0.025563, 0.035591, 0.007018], [0., -0.12475, 0.12202, 0.39377, -0.13591, 0.44293, 0.00341, 0.32967, 0.14563, -0.11359, -0.055964, 0.077901, 0.015359], [0., -0.19991, 0.19555, 0.63108, -0.21776, 0.18676, -1.2152, -0.45883, -0.20267, 0.15808, 0.077878, -0.10842, -0.021378], [0., 0.039183, -0.038327, -0.12368, 0.042683, 0.40494, -0.14642, 0.45844, 0.00530, 0.30698, 0.15121, -0.21050, -0.041508], [0., 0.063362, -0.061979, -0.20000, 0.06902, 0.65480, -0.23675, 0.19319, -1.2055, -0.43689, -0.21522, 0.29961, 0.059080], [0., -0.011856, 0.011596, 0.037422, -0.012915, -0.12252, 0.044302, 0.38539, -0.14162, 0.51206, -0.01359, 0.57415, 0.11321], [0., -0.020725, 0.020273, 0.065420, -0.022576, -0.21417, 0.077448, 0.67372, -0.24758, 0.11995, -1.1771, -0.81929, -0.16154], [0., 0.0051199, -0.0050080, -0.016161, 0.0055772, 0.052912, -0.019131, -0.16643, 0.061155, 0.52691, -0.19397, 0.15958, -0.66067], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    Bp = ca.DM([[0., -15., 0., 0.], [0., 0.62472, 0., -0.0050820], [0., 1.0000, 0., 0.], [0., -0.19714, 0., 0.0080921], [0., -0.31562, 0., -0.0092612], [0., 0.062185, 0., -0.020271], [0., 0.099656, 0., 0.028214], [0., -0.019532, 0., 0.054779], [0., -0.031585, 0., -0.077964], [0., 0.0059099, 0., -0.14940], [0., 0.010331, 0., 0.21319], [0., -0.0025522, 0., -0.70055], [0., 0., 0., 1.0000]])

    Bp_d = ca.DM([15., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    # ca.DM([0., -0.028882, 0., 0., 0., -0.10716, 0.13604])
    operating_point = ca.DM([0., 0., 0., 0.047574, 0., 0.047574, 0., 0.047574, 0., 0.047574, -0.10000, 0.092988, 0.]) * 1
    # Un-constraint
    #lower_bounds_input = None
    lower_bounds_slew_rate = None
    #lower_bounds_states = None
    #upper_bounds_input = None
    #upper_bounds_slew_rate = None
    #upper_bounds_states = None

    # Actual constraints
    #lower_bounds_input = ca.DM([0, 0, 0, 0])
    # lower_bounds_slew_rate = ca.DM([-ca.inf, -ca.inf, -ca.inf, -ca.inf])
    #lower_bounds_states = ca.DM([0, 0, 0, 0, 0, 0, 0])
    #upper_bounds_input = ca.DM([2, 2, ca.inf, ca.inf])
    #upper_bounds_slew_rate = ca.DM([1, 1, ca.inf, ca.inf])
    #upper_bounds_states = ca.DM([3, 1, 1, 1, 1, 1, 2])

    # size1 and size2 represent the num of rows and columns in the Casadi lib, respectively
    num_states = Ap.size1()
    num_inputs = Bp.size2()

    # TODO: check if this is the right thing to put here
    # TODO: make sure when we feed the disturbance it's also lifted to Dd from D?
    disturbance_array = (np.random.rand(pred_horizon * num_inputs) - 0.5) * disturb_magnitude

    # Initial conditions here are needed for the right dimensions in the MPC
    [x0, u0] = set_preismann_initial_conditions()
    [Q, R] = set_preismann_weight_matrices()

    initial_state_space_model = {"system_model": Ap,
                                 "b_matrix": Bp,
                                 "b_disturbance": Bp_d,
                                 "Q": Q,
                                 "R": R,
                                 "x0": x0,
                                 "u0": u0,
                                 "operating_point": operating_point,
                                 "num_states": num_states,
                                 "num_inputs": num_inputs,
                                 "sim_type": simulation_type,
                                 "disturbance_array": disturbance_array,
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
                                   network_df):

    pump1 = Links(sim)[pump_ids[0]]
    pump2 = Links(sim)[pump_ids[1]]
    tank1 = Nodes(sim)[tank_ids[0]]
    tank2 = Nodes(sim)[tank_ids[1]]

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
    operating_point = complete_model["operating_point"]

    # start the simulation with the pumps closed
    # https://pyswmm.readthedocs.io/en/stable/reference/nodes.html
    # use these functions to set how much the pump is open for
    pump1.target_setting = int(0)
    pump2.target_setting = int(0)

    # x initial conditions
    states = [tank1.initial_depth,
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
    ref = set_preismann_ref(prediction_horizon, complete_model["num_states"])

    # This disturbance is delta_disturbance between consecutive ones
    disturbance = ca.DM.zeros(prediction_horizon, 1)

    # To make the simulation precise,
    # make sure that the flow metrics are in Cubic Meters per Second [CMS]
    for idx, step in enumerate(sim):

        network_df = network_df.append(network_elements, ignore_index=True)

        user_key_input = input("press s key to step, or \'r\' to run all steps, or q to quit")

        if user_key_input == "r":
            print('R')

        mpc_model.plot_progress(options={'drawU': 'U'})
        mpc_model.step(states, control_input)

        control_input = control_input + mpc_model.get_next_control_input_change()
        pump1.target_setting = control_input[0]
        pump2.target_setting = control_input[1]

        sim.step_advance(time_step)

        # tank1.depth, hp1.depth, hp2.depth ... hp5.depth, tank2.depth
        states = [tank1.depth,
                  junction_n1.depth,
                  junction_n2.depth,
                  junction_n3.depth,
                  junction_n4.depth,
                  junction_n5.depth,
                  tank2.depth
                  ]

    return network_df
