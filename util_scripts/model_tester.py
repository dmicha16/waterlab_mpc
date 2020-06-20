import pandas as pd
import casadi as ca
import matplotlib.pyplot as plt


def run_euler_model(steps):
    # System matrices w/ backflow
    A = ca.DM([[1., 0., 0., 0., 0., 0., 0.], [0., 0.84717, 0.0013996, 0., 0., 0., 0.],
               [0., 0.15283, 0.84577, 0.0013996, 0., 0., 0.], [0., 0., 0.15283, 0.84577, 0.0013996, 0., 0.],
               [0., 0., 0., 0.15283, 0.84577, 0.0013996, 0.], [0., 0., 0., 0., 0.15283, 0.80758, 0.039586],
               [0., 0., 0., 0., 0., 0.19102, 0.96041]])

    system = "w/ backflow"

    B = ca.DM(
        [[-0.08413333333, 0, -0.08413333333, 0], [0.03155000000, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
         [0, 0, 0, 0], [0, -0.06310000000, 0, -0.06310000000]])

    op_gain = 0
    operating_point = ca.DM([0., 0.028683, 0., 0., 0., -0.039586, 0.010903]) * op_gain

    # *****************

    # System matrices w/o backflow

    A = ca.DM([[1., 0., 0., 0., 0., 0., 0.], [0., 0.84717, 0.0013996, 0., 0., 0., 0.],
               [0., 0.15283, 0.84577, 0.0013996, 0., 0., 0.], [0., 0., 0.15283, 0.84577, 0.0013996, 0., 0.],
               [0., 0., 0., 0.15283, 0.84577, 0.0013996, 0.], [0., 0., 0., 0., 0.15283, 0.95618, 0.],
               [0., 0., 0., 0., 0., 0.042425, 1.]])

    system = "w/o backflow"

    B = ca.DM(
        [[-0.08413333333, 0, -0.08413333333, 0], [0.03155000000, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
         [0, 0, 0, 0], [0, -0.06310000000, 0, -0.06310000000]])

    op_gain = 1
    operating_point = ca.DM([0., 0.028684, 0., 0., 0., -0.025064, -0.0036202]) * op_gain

    # *****************

    # initial conditions
    x0 = ca.DM.zeros(7, 1)

    # set the control input to always on
    u = ca.DM.ones(4, 1)

    # turn the 2nd pump off
    u[0] = 1
    u[1] = 0
    u[2] = 0
    u[3] = 0
    print(u)
    x = x0

    states_df = pd.DataFrame()

    temp_x = []

    for step in range(steps):
        # state space model
        x = A @ x + B @ u + operating_point

        temp_x = x.elements()
        states_row = pd.Series(temp_x)

        states_df = states_df.append(states_row, ignore_index=True)

        temp_x.clear()

    # plotting
    for idx in range(len(states_df.keys())):
        if idx != 0 and idx != 6:
            states_df[idx].plot()

    plt.grid()
    plt.title(f"type: {system}, OP gain = {op_gain}, u = {u}")
    plt.xlabel("Steps")
    plt.ylabel("depth [m]")
    # plt.legend(["State 1", "State 2", "State 3", "State 4", "State 5", "tank2"])
    plt.legend(["State 1", "State 2", "State 3", "State 4", "State 5"])
    plt.show()


def run_priessman_model(steps):
    A = ca.DM([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., -0.57690, 0.32141, 0.11616, -0.00806, -1.3343, 0.01643, 1.5745, -0.02942, -1.9671, 0.04957, -1.0914, 0.90771], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], [0., 1.3295, -0.25344, 0.63522, 0.04337, 1.3427, -0.01652, -1.5843, 0.02961, 1.9795, -0.04988, 1.0983, -0.91339], [0., 5.8510, -1.1153, 5.8806, -1.3277, -0.19762, 0.002434, 0.23319, -0.004359, -0.29133, 0.007338, -0.16164, 0.13443], [0., -0.92533, 0.17639, 0.52445, -0.06726, -0.6130, 0.05206, 1.6107, -0.03011, -2.0124, 0.05071, -1.1166, 0.92859], [0., -3.7105, 0.70732, 2.1033, -0.26973, 6.1943, -1.3306, -0.39037, 0.007293, 0.48773, -0.01229, 0.27060, -0.22506], [0., 0.66951, -0.12762, -0.37947, 0.04867, 1.6334, -0.07625, -0.8962, 0.06604, 2.0594, -0.05189, 1.1426, -0.95026], [0., 2.3410, -0.44621, -1.3269, 0.17023, 5.7112, -0.26658, 6.3626, -1.3358, -0.62377, 0.01572, -0.34608, 0.28783], [0., -0.50862, 0.096954, 0.28827, -0.03697, -1.2409, 0.05792, 1.9303, -0.09067, -1.3642, 0.08856, -1.1742, 0.97652], [0., -1.4643, 0.27913, 0.82986, -0.10640, -3.5725, 0.16679, 5.5571, -0.26105, 6.5851, -1.3447, 0.40035, -0.33296], [0., 0.44672, -0.085151, -0.25320, 0.032479, 1.0899, -0.050867, -1.6953, 0.079624, 2.6426, -0.12460, 2.1912, -1.5796], [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    system = "Preissmann"

    B = ca.DM([[-11.886, 0., -11.886, 0.], [0.21376, 1.1940, 0., 268.53], [1.0000, 0., 0., 0.], [-0.14457, -1.2015, 0., -270.20], [-0.63619, 0.17683, 0., 39.769], [0.10062, 1.2215, 0., 274.71], [0.40347, -0.29604, 0., -66.578], [-0.072797, -1.2500, 0., -281.12], [-0.25455, 0.37861, 0., 85.147], [0.055305, 1.2845, 0., 288.88], [0.15921, -0.43798, 0., -98.503], [-0.048574, -2.4988, 0., -293.05], [0., 1.0000, 0., 0.]])

    op_gain = 0
    operating_point = ca.DM(
        [0., 1.8571, 0., -1.7729, -1.9879, 1.7396, -1.1978, -1.7393, -1.2053, 1.7608, -1.7156, -2.8334, 0.]) * op_gain

    # *****************

    # initial conditions
    x0 = ca.DM.zeros(13, 1)

    # set the control input to always on
    u = ca.DM.ones(4, 1)

    # turn the 2nd pump off
    u[0] = 0
    u[1] = 1
    u[2] = 0
    u[3] = 0
    print(u)
    x = x0

    states_df = pd.DataFrame()

    temp_x = []

    for step in range(steps):
        # state space model
        x = A @ x + B @ u + operating_point

        temp_x = x.elements()
        states_row = pd.Series(temp_x)

        states_df = states_df.append(states_row, ignore_index=True)

        temp_x.clear()

    # plotting
    for idx in range(len(states_df.keys())):
        if idx != 0 and idx != 13:
            states_df[idx].plot()

    plt.grid()
    plt.title(f"type: {system}, OP gain = {op_gain}, u = {u}")
    plt.xlabel("Steps")
    plt.ylabel("depth [m]")
    # plt.legend(["State 1", "State 2", "State 3", "State 4", "State 5", "tank2"])
    plt.legend(
        ["State 1", "State 2", "State 3", "State 4", "State 5", "State 6", "State 7", "State 8", "State 9", "State 10",
         "State 11", "State 12", "State 13"])
    plt.show()


if __name__ == "__main__":

    model_type = "priessman"
    steps = 12

    if model_type == "euler":
        run_euler_model(steps)
    elif model_type == "priessman":
        run_priessman_model(steps)
