import casadi as ca
import numpy as np

from controller import mpco
from pyswmm import Simulation, Nodes, Links
from enum import Enum
import pandas as pd


def set_preismann_inital_conditions():
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
    Q = ca.DM(np.identity(7))
    R = ca.DM(np.identity(4))

    return [Q, R]


def set_preismann_ref(prediction_horizon, num_states):
    """
        Create an arbitrary reference for the Preismann model
        :param prediction_horizon: Length of the prediction horizon.
        :param num_states: The number of states in the model
        :return: Reference corresponding to the Preismann model
        """

    ref = ca.DM.ones(prediction_horizon * num_states, 1)

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

    Ap = ca.DM([[1, 0, 0], [0, 0.5, 0], [0, 0, 0.7]])
    Bp = ca.DM([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    Bp_d = ca.DM([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 1]])

    # size1 and size2 represent the num of rows and columns in the Casadi lib, respectively
    num_states = Ap.size1()
    num_inputs = Bp.size2()

    # TODO: check if this is the right thing to put here
    disturbance_array = (np.random.rand(pred_horizon * num_inputs) - 0.5) * disturb_magnitude

    [x0, u0] = set_preismann_inital_conditions()
    [Q, R] = set_preismann_weight_matrices()

    initial_state_space_model = {"system_model": Ap,
                                 "b_matrix": Bp,
                                 "b_disturbance": Bp_d,
                                 "x0": x0,
                                 "u0": u0,
                                 "Q": Q,
                                 "R": R,
                                 "num_states": num_states,
                                 "num_inputs": num_inputs,
                                 "sim_type": simulation_type,
                                 "disturbance_array": disturbance_array
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

    # TODO add constraints to model
    mpc_model = mpco.MpcObj(state_space_model["system_model"],
                            state_space_model["b_matrix"],
                            control_horizon,
                            prediction_horizon,
                            state_space_model["Q"],
                            state_space_model["R"],
                            initial_control_signal=state_space_model["u0"],
                            ref=ref,
                            input_matrix_d=state_space_model["b_disturbance"],
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

    mpc_model = complete_model["mpc_model"]
    operating_point = complete_model["operating_point"]

    # start the simulation with the pumps closed
    # https://pyswmm.readthedocs.io/en/stable/reference/nodes.html
    # use these functions to set how much the pump is open for
    pump1.target_setting = int(0)
    pump2.target_setting = int(0)

    # x initial conditions
    states = [tank1.depth,
              junction_n1.depth,
              junction_n2.depth,
              junction_n3.depth,
              junction_n4.depth,
              junction_n5.depth,
              tank2.depth
              ]

    # u_prev, initial control input
    control_input = ca.DM.zeros(complete_model["num_inputs"], 1)

    # zeros for now
    # TODO: make sure to add a proper reference
    ref = set_preismann_ref(prediction_horizon, complete_model["num_states"])

    # This disturbance is delta_disturbance between consecutive ones
    disturbance = []

    # To make the simulation precise,
    # make sure that the flow metrics are in Cubic Meters per Second [CMS]
    for idx, step in enumerate(sim):

        # TODO: finish filling of dataframe
        network_df = network_df.append(pd.Series([tank1.volume, tank1.depth, tank1.flooding, tank1.total_inflow,
                                                  pump1.flow], index=network_df.columns), ignore_index=True)

        user_key_input = input("press s key to step, or \'r\' to run all steps, or q to quit")

        if user_key_input == "r":
            print('R')

        mpc_model.plot_progress(options={'drawU': 'U'})
        mpc_model.step(states, control_input)

        # TODO: don't forget to add operating point
        control_input = control_input + mpc_model.get_next_control_input_change()
        pump1.target_setting = control_input[0]
        pump2.target_setting = control_input[1]

        sim.step_advance(time_step)

        # tank1.depth, hp1.depth, hp2.depth ... hp5.depth, tank2.depth
        # this here is a python stl list, you cannot substruct like this safely
        states = [tank1.depth,
                  junction_n1.depth,
                  junction_n2.depth,
                  junction_n3.depth,
                  junction_n4.depth,
                  junction_n5.depth,
                  tank2.depth
                  ] - operating_point

    return network_df