"""
Module for Level control
"""

import casadi as ca
import mpc


def gen_mpc_solver(A, B, Hu, Hp, Q, R):
    # # The wrong way of declaring inputs
    # Solver inputs
    # x0 = ca.SX.sym('x0', A.shape[0], 1)
    # u_prev = ca.SX.sym('u_prev', B.shape[1], 1)
    # ref = ca.SX.sym('ref', Hp * A.shape[0], 1)
    # input_variables = ca.vertcat(ca.vertcat(x0, u_prev), ref)

    # Solver inputs
    input_variables = ca.SX.sym('i', A.shape[0] + B.shape[1] + Hp * A.shape[0], 1)
    x0 = input_variables[0:A.shape[0], :]
    u_prev = input_variables[A.shape[0]:(A.shape[0] + B.shape[1]), :]
    ref = input_variables[(A.shape[0] + B.shape[1]):(A.shape[0] + B.shape[1] + Hp * A.shape[0]), :]

    # Solver outputs
    dU = ca.SX.sym('dU', B.shape[1] * Hu, 1)

    # To formulate a MPC optimization problem we need to describe:
    # Z = psi x(k) + upsilon u(k-1) + Theta dU(x) (Assuming no disturbance)
    psi = mpc.gen_psi(A, Hp)
    upsilon = mpc.gen_upsilon(A, B, Hp)
    theta = mpc.gen_theta(upsilon, B, Hu)
    predicted_states = mpc.gen_predicted_states(psi, x0, upsilon, u_prev, theta, dU)

    # Cost function:
    # Cost = (Z - T)' * Q * (Z - T) + dU' * R * dU
    error = predicted_states - ref  # e = (Z - T)
    quadratic_cost = error.T @ Q @ error + dU.T @ R @ dU
    quadratic_problem = {'x': dU, 'p': input_variables, 'f': quadratic_cost}
    print(quadratic_cost)

    mpc_solver = ca.qpsol('mpc_solver', 'qpoases', quadratic_problem)
    print(mpc_solver)

    return mpc_solver


# def mpc_version_1(pump,tank):
def on_off(pump, tank):
    if tank.depth >= tank.full_depth * 0.5:
        pump.target_setting = 5  # [0,1]

    elif tank.depth <= tank.full_depth * 0.1:
        pump.target_setting = 0  # [0,1]

    return pump
