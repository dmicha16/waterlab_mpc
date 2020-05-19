import casadi as ca
import numpy as np

from controller import mpco


def set_custom_model_inital_conditions():
    """
    Set initial conditions for the Custom model
    :return: x0 and u0
    """

    x0 = ca.DM([[10], [11], [12]])
    u0 = ca.DM([-1, -3, -6])

    return [x0, u0]


def set_custom_model_weight_matrices():
    """
    Set the Q and R weight matrices for the Custom model
    :return: Q and R matrices
    """

    Q = ca.DM([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    R = ca.DM([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])

    return [Q, R]


def set_custom_model_ref(prediction_horizon, num_states):
    """
        Create an arbitrary reference for the Custom model
        :param prediction_horizon: Length of the prediction horizon.
        :param num_states: The number of states in the model
        :return: Reference corresponding to the Custom model
        """

    ref = ca.DM.ones(prediction_horizon * num_states, 1)
    for state in range(num_states):
        ref[state::num_states] = ref[state::num_states] + state - 2

    return ref


def make_custom_model_model(simulation_type, pred_horizon, disturb_magnitude):
    """
    Construct the Custom Model model based state space model, using the prediction horizon and the
    disturbance magnitude
    :param simulation_type: This is to denote in the state_space_mode|l which type of simulation it is.
    :param pred_horizon: Length of the prediction horizon
    :param disturb_magnitude: Magnitude of the disturbances
    :return: Entire state space model of the Custom Model model with both the initial conditions and with the
    weight matrices.
    """

    A = ca.DM([[1, 0, 0], [0, 0.5, 0], [0, 0, 0.7]])
    B = ca.DM([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    B_d = ca.DM([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 1]])

    # size1 and size2 represent the num of rows and columns in the Casadi lib, respectively
    num_states = A.size1()
    num_inputs = B.size2()

    # these are just random values for the custom model, numbers are arbitrary
    rand_var = 0.00
    const_var = 1.00
    # rand_A = np.random.rand(A.size1(), A.size2())

    A * const_var + (np.random.rand(A.size1(), A.size2()) - 0.5) * rand_var
    B * const_var + (np.random.rand(B.size1(), B.size2()) - 0.5) * rand_var
    B_d * const_var + (np.random.rand(B_d.size1(), B_d.size2()) - 0.5) * rand_var

    # TODO: check if this is the right thing to put here
    disturbance_array = (np.random.rand(pred_horizon * num_inputs) - 0.5) * disturb_magnitude

    [x0, u0] = set_custom_model_inital_conditions()
    [Q, R] = set_custom_model_weight_matrices()

    initial_state_space_model = {"system_model": A,
                                 "b_matrix": B,
                                 "b_disturbance": B_d,
                                 "x0": x0,
                                 "u0": u0,
                                 "Q": Q,
                                 "R": R,
                                 "num_states": num_states,
                                 "num_inputs": num_inputs,
                                 "sim_type": simulation_type,
                                 "disturbance_array": disturbance_array
                                 }

    print("Custom Model is constructed.")

    return initial_state_space_model

def make_custom_mpc_model(state_space_model, prediction_horizon, control_horizon):
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

    ref = set_custom_model_ref(prediction_horizon, num_states)

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
