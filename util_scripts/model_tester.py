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


def run_priessman_model():
    pass


if __name__ == "__main__":

    model_type = "euler"
    steps = 700

    if model_type == "euler":
        run_euler_model(steps)
    elif model_type == "priessman":
        run_priessman_model()
