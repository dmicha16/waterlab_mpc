import casadi as ca
import numpy as np

from controller import mpco
from pyswmm import Simulation, Nodes, Links
from enum import Enum


class PumpSetting(Enum):
    CLOSED = 0
    OPEN = 1


# DEPRECATED
def set_euler_initial_conditions():
    """
    Set initial conditions for the Euler model
    :return: x0 and u0
    """

    # TODO: figure out how to get the initial conditions from pyswmm here
    x0 = ca.DM.zeros(7, 1)
    u0 = ca.DM.zeros(3, 1)

    return [x0, u0]


def set_euler_weight_matrices():
    """
    Set the Q and R weight matrices for the Euler model
    :return: Q and R matrices
    """

    # initially to ones to run the code
    # TODO: add proper Q and R matrices @Casper
    Q = ca.DM.ones(7, 7)
    R = ca.DM.ones(3, 3)

    return [Q, R]


def set_euler_ref(prediction_horizon, num_states):
    """
        Create an linear arbitrary reference for the Euler model
        :param prediction_horizon: Length of the prediction horizon.
        :param num_states: The number of states in the model
        :return: Reference corresponding to the Euler model
        """

    ref = ca.DM.ones(prediction_horizon * num_states, 1)

    return ref


def make_euler_model(simulation_type, pred_horizon, disturb_magnitude):
    """
    Construct the Euler model based state space model, using the prediction horizon and the
    disturbance magnitude
    :param simulation_type: This is to denote in the state_space_model which type of simulation it is.
    :param pred_horizon: Length of the prediction horizon
    :param disturb_magnitude: Magnitude of the disturbances
    :return: Entire state space model of the Euler model with both the initial conditions and with the
    weight matrices.
    """

    Ap = ca.DM([[1, 0, 0, 0, 0, 0, 0],
                [0, 81.57745346, -76.82745346, 0, 0, 0, 0],
                [0, -80.57745346, 158.4049069, -76.82745346, 0, 0, 0],
                [0, 0, -80.57745346, 158.4049069, -76.82745346, 0, 0],
                [0, 0, 0, -80.57745346, 158.4049069, -76.82745346, 0],
                [0, 0, 0, 0, -80.57745346, 696.1970811, -76.82745346],
                [0, 0, 0, 0, 0, -618.3696276, 615.6196276]])

    Bp = ca.DM([[-24, 0, -24, 0], [90, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, -144, 0, -144]])
    Bp_d = ca.DM([24, 0, 0, 0, 0, 0, 0])

    operating_point = ca.DM([0, -152.1549069, 0, 0, 0, -1229.239255, 1381.394162])

    # size1 and size2 represent the num of rows and columns in the Casadi lib, respectively
    num_states = Ap.size1()
    num_inputs = Bp.size2()

    # TODO: check if this is the right thing to put here
    # TODO: make sure when we feed the disturbance it's also lifted to Dd from D?
    disturbance_array = (np.random.rand(pred_horizon * num_inputs) - 0.5) * disturb_magnitude

    # Initial conditions here are needed for the right dimensions in the MPC
    [x0, u0] = set_euler_initial_conditions()
    [Q, R] = set_euler_weight_matrices()

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
                                 "disturbance_array": disturbance_array
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
                            state_space_model["x0"],
                            state_space_model["u0"],
                            ref,
                            state_space_model["b_disturbance"],
                            state_space_model["disturbance_array"])

    state_space_model["mpc_model"] = mpc_model

    return state_space_model


def run_euler_model_simulation(time_step, complete_model, prediction_horizon, sim, pump_ids, tank_ids, junction_ids):

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
    pump1.current_setting = PumpSetting.CLOSED
    pump2.current_setting = PumpSetting.CLOSED

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
    ref = set_euler_ref(prediction_horizon, complete_model["num_states"])

    # This disturbance is delta_disturbance between consecutive ones
    disturbance = []

    # To make the simulation precise,
    # make sure that the flow metrics are in Cubic Meters per Second [CMS]
    for idx, step in enumerate(sim):

        # TODO: finish filling of dataframe
        # network_df = network_df.append(pd.Series([tank1.volume, tank1.depth, tank1.flooding, tank1.total_inflow,
        #                                           pump1.flow], index=network_df.columns), ignore_index=True)

        user_key_input = input("press s key to step, or \'r\' to run all steps, or q to quit")
        # TODO: try non-blocking user input to fix printing like this
        if user_key_input == "r":
            mpc_model.plot_progress(options={'drawU': 'U'}, ignore_inputs=[1, 2])
            mpc_model.step(states, control_input, ref, disturbance)

        elif user_key_input == "s":
            mpc_model.plot_progress(options={'drawU': 'U'}, ignore_inputs=[1, 2])
            mpc_model.plot_step({'drawU': 'both'})
            # user_key_input = input("press s key to step, or \'r\' to run all steps, or q to quit")

        elif user_key_input == "rw":
            # TODO: make functionality in MPC to not plot each step, only the last file
            print("Running the simulation without plots.")
            mpc_model.step(states, control_input, ref, disturbance)

        elif user_key_input == "q":
            print("Quitting.")
            break

        # TODO: don't forget to add operating point
        control_input = control_input + mpc_model.get_next_control_input_change()
        pump1.current_setting = control_input[0]
        pump2.current_setting = control_input[1]

        # tank1.depth, hp1.depth, hp2.depth ... hp5.depth, tank2.depth
        states = [tank1.depth,
                  junction_n1.depth,
                  junction_n2.depth,
                  junction_n3.depth,
                  junction_n4.depth,
                  junction_n5.depth,
                  tank2.depth
                  ]

        # TODO: make sure to sync the disturbances in with the length of the time step,
        #  or we need to use the disturbance multiple times
        #
        sim.step_advance(time_step)
