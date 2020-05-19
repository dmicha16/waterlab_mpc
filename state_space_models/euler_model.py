import casadi as ca
import numpy as np

from controller import mpco
from pyswmm import Simulation, Nodes, Links
from enum import Enum

class PumpSetting(Enum):
    CLOSED = 0
    OPEN = 1


def set_euler_initial_conditions():
    """
    Set initial conditions for the Euler model
    :return: x0 and u0
    """

    # TODO: figure out how to get the initial conditions from pyswmm here
    x0 = ca.DM([[10], [11], [12]])
    u0 = ca.DM([-1, -3, -6])

    return [x0, u0]


def set_euler_weight_matrices():
    """
    Set the Q and R weight matrices for the Euler model
    :return: Q and R matrices
    """

    Q = ca.DM([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    R = ca.DM([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])

    return [Q, R]


def set_euler_ref(prediction_horizon, num_states):
    """
        Create an arbitrary reference for the Euler model
        :param prediction_horizon: Length of the prediction horizon.
        :param num_states: The number of states in the model
        :return: Reference corresponding to the Euler model
        """

    ref = ca.DM.ones(prediction_horizon * num_states, 1)
    for state in range(num_states):
        ref[state::num_states] = ref[state::num_states] + state - 2

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

    Ap = ca.DM([[1, 0, 0], [0, 0.5, 0], [0, 0, 0.7]])
    Bp = ca.DM([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    Bp_d = ca.DM([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 1]])

    # size1 and size2 represent the num of rows and columns in the Casadi lib, respectively
    num_states = Ap.size1()
    num_inputs = Bp.size2()

    # TODO: check if this is the right thing to put here
    disturbance_array = (np.random.rand(pred_horizon * num_inputs) - 0.5) * disturb_magnitude

    [x0, u0] = set_euler_initial_conditions()
    [Q, R] = set_euler_weight_matrices()

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
                            state_space_model["B_matrix"],
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


def run_euler_model_simulation(complete_model, prediction_horizon, sim, pump_ids, tank_ids):

    pump1 = Links(sim)[pump_ids[0]]
    pump2 = Links(sim)[pump_ids[1]]
    tank1 = Nodes(sim)[tank_ids[0]]
    tank2 = Nodes(sim)[tank_ids[1]]

    mpc_model = complete_model["mpc_model"]

    # start the simulation with the pumps closed
    # https://pyswmm.readthedocs.io/en/stable/reference/nodes.html
    # TODO: here I could make a function which prints the information about the topo
    # use these functions to set how much the pump is open for
    pump1.current_setting = PumpSetting.CLOSED
    pump2.current_setting = PumpSetting.CLOSED

    for idx, step in enumerate(sim):

        # TODO: Need @casper for this to figure out at the Euler model
        states = [tank1.depth, tank2.depth]

        # TODO: finish filling of dataframe
        # network_df = network_df.append(pd.Series([tank1.volume, tank1.depth, tank1.flooding, tank1.total_inflow,
        #                                           pump1.flow], index=network_df.columns), ignore_index=True)

        user_key_input = input("press s key to step, or \'r\' to run all steps, or q to quit")
        # TODO: try non-blocking user input to fix printing like this
        if user_key_input == "r":
            mpc_model.plot_progress(options={'drawU': 'U'}, ignore_inputs=[1, 2])
            mpc_model.step(x, u, ref, disturbance)

        elif user_key_input == "s":
            mpc_model.plot_progress(options={'drawU': 'U'}, ignore_inputs=[1, 2])
            mpc_model.plot_step({'drawU': 'both'})
            user_key_input = input("press s key to step, or \'r\' to run all steps, or q to quit")

        elif user_key_input == "rw":
            # TODO: make functionality in MPC to not plot each step, only the last file
            print("Running the simulation without plots.")
            mpc_model.step(x, u, ref, disturbance)

        elif user_key_input == "q":
            print("Quitting.")
            break

    u = u + mpc_model.get_next_control_input_change()
    pump1.current_setting = u[0]
    mpc_model.plot_progress(options={'drawU': 'U'}, ignore_inputs=[1, 2])

    sim.step_advance(time_step)
