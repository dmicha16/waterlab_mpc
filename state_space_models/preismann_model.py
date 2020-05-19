import casadi as ca
import numpy as np


def set_preismann_inital_conditions():
    """
    Set initial conditions for the Preismann model
    :return: x0 and u0
    """

    x0 = ca.DM([[10], [11], [12]])
    u0 = ca.DM([-1, -3, -6])

    return [x0, u0]


def set_preismann_weight_matrices():
    """
    Set the Q and R weight matrices for the Preismann model
    :return: Q and R matrices
    """

    Q = ca.DM([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    R = ca.DM([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])

    return [Q, R]


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
