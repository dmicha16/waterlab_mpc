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

    Ap = ca.DM([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0.398711853433386, 0.763137771602704, 0.593909306864655, 0.270504977946570, -0.227225389658993,
                 -0.0991252142993026, 0.0826466576723167, 0.0365075850711874, -0.0284785796643008, -0.0140271641249692,
                 0.0195302497250856, 0.00385166315464575], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0.395506320581628, -0.386856463693640, 0.403616812756526, -0.0319154645445088, 0.361802489115548,
                 0.157833371180140, -0.131595181805166, -0.0581296622273410, 0.0453453772242729, 0.0223349287826617,
                 -0.0310972861535790, -0.00613285917861918],
                [0, 0.633174849184573, -0.619327101258660, 0.00761193962713614, -1.27258311816019, -0.414083382943246,
                 -0.180640482710269, 0.150610843485689, 0.0665294681727803, -0.0518978386460681, -0.0255623527929001,
                 0.0355908813184414, 0.00701906468914902],
                [0, -0.124749828967759, 0.122021508050447, 0.393767377906814, -0.135894694577357, 0.442932100079981,
                 0.00341422530542563, 0.329656497596236, 0.145619471727233, -0.113593811208574, -0.0559507901091301,
                 0.0779011989459815, 0.0153633047148172],
                [0, -0.199922048088868, 0.195549685335779, 0.631045199124552, -0.217782628554447, 0.186733265685896,
                 -1.21524847189662, -0.458808589717605, -0.202669945673043, 0.158097342844089, 0.0778710666731564,
                 -0.108421158042797, -0.0213823061914603],
                [0, 0.0391830288992562, -0.0383260828157700, -0.123679516643623, 0.0426835514640771, 0.404943426780458,
                 -0.146420550620764, 0.458445559660545, 0.00529196257857980, 0.306964581928667, 0.151195832868854,
                 -0.210512427673438, -0.0415162618337312],
                [0, 0.0633604136603723, -0.0619747000016880, -0.199994373992960, 0.0690208887171826, 0.654808567653776,
                 -0.236767470926352, 0.193185080261529, -1.20547435892170, -0.436889643986966, -0.215190603356763,
                 0.299613391887839, 0.0590883310975056],
                [0, -0.0118554235389751, 0.0115961414197720, 0.0374211257806489, -0.0129145600779189,
                 -0.122521815405072, 0.0443017727619329, 0.385387687142440, -0.141614880257334, 0.512058901208463,
                 -0.0135731464662892, 0.574153163163703, 0.113231761744473],
                [0, -0.0207245259097235, 0.0202712735244598, 0.0654160594315744, -0.0225760078556121,
                 -0.214181005808950, 0.0774441532546661, 0.673698167021573, -0.247557689140917, 0.119961177129556,
                 -1.17710656151850, -0.819281178851439, -0.161574745550930],
                [0, 0.00511995261831495, -0.00500797752432258, -0.0161609064653872, 0.00557735752484492,
                 0.0529129885171636, -0.0191324229541664, -0.166435782860429, 0.0611586312865357, 0.526928481358642,
                 -0.193986486027224, 0.159580453709580, -0.660703554048525], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    Bp = ca.DM([[-15, 0, -15, 0], [0.624714859422507, -0.00508195629848902, 0, 0], [1, 0, 0, 0],
                [-0.197141955900345, 0.00809180893530043, 0, 0], [-0.315608933914345, -0.00926108503634281, 0, 0],
                [0.0621821295921914, -0.0202706311601780, 0, 0], [0.0996520701107426, 0.0282122141171264, 0, 0],
                [-0.0195309620942072, 0.0547773312059149, 0, 0], [-0.0315822914215166, -0.0779622475620642, 0, 0],
                [0.00590939072999763, -0.149400101120539, 0, 0], [0.0103302358529743, 0.213184736790626, 0, 0],
                [-0.00255206407778098, -0.700554859422507, 0, -20], [0, 1, 0, 0]])

    Bp_d = ca.DM([11.88589540, 0., 0., 0., 0., 0., -0., 0., 0., 0., 0., 0., 0.])

    # ca.DM([0., -0.028882, 0., 0., 0., -0.10716, 0.13604])
    operating_point = ca.DM(
        [[0], [-0.720807290080096], [0], [0.280751386180611], [1.35401816584457], [-0.234071271146695],
         [1.21038704266329], [0.471105981350287], [0.481049472806078], [-1.23341779768537], [2.82662429229865],
         [0], [0]]) * 1

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
