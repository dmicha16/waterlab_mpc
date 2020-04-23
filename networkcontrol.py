"""
Module for Level control
"""

import casadi as ca
import mpc

def gen_mpc_solver(A, B, Hu, Hp, Q, R):

    # Solver inputs
    x0 = ca.SX(A.shape[0], 1)
    u_prev = ca.SX(B.shape[1], 1)
    ref = ca.SX(Hp * A.shape[0], 1)

    # Solver outputs
    dU = ca.SX.sym('dU', B.shape[1] * Hu, 1)

    # To formulate a MPC optimization problem we need to describe:
    # Z = psi x(k) + upsilon u(k-1) + Theta dU(x) (Assuming no disturbance)
    psi = mpc.gen_psi(A, Hp)
    upsilon = mpc.gen_upsilon(A, B, Hp)
    theta = mpc.gen_theta(upsilon, B, Hu)
    predicted_states = mpc.gen_predicted_states(psi, x0, upsilon, u_prev, theta, dU)

    # Cost function:
    # Cost = (Z - T)' * Q * (Z - T) + dU' * R * dU;
    error = predicted_states - ref # e = (Z - T)
    quadratic_cost = error.T @ Q @ error + dU.T @ R @ dU
    quadratic_problem = {'x': dU, 'f': quadratic_cost}

    mpc_solver = ca.qpsol('mpc_solver', 'qpoases', quadratic_problem)
    print(mpc_solver)

    # U = ca.SX(B.shape[1], Hp)
    # for t in range(0, Hp):
    #     if t < Hu:
    #         U[:, t] = ca.sum2(dU[:, 0:t + 1]) + u_prev
    #     else:
    #         U[:, t] = U[:, t - 1]
    return mpc_solver



# def mpc_version_1(pump,tank):
def on_off(pump, tank):

    if tank.depth >= tank.full_depth * 0.5:
        pump.target_setting = 5  # [0,1]

    elif tank.depth <= tank.full_depth * 0.1:
        pump.target_setting = 0  # [0,1]

    return pump