"""
Module for Level control

"""
import casadi as ca
import  mpc
def gen_mpc_solver(A, B, Hu, Hp, Q, R)

    U = ca.SX(B.shape[1], Hp)
    # x0, u_prev, ref as SX

    for t in range(0, Hp):
        if t < Hu:
            U[:, t] = ca.sum2(dU[:, 0:t + 1]) + u_prev
        else:
            U[:, t] = U[:, t - 1]

    dU = ca.SX.sym('dU', B.shape[1] * Hu, 1)

    psi = mpc.gen_psi(A, Hp)
    upsilon = mpc.gen_upsilon(A, B, Hp)
    theta = mpc.gen_theta(upsilon, B, Hu)

    predicted_states = mpc.gen_predicted_states(psi, x0, upsilon, u_prev, theta, dU)
    error = predicted_states - ref
    quadratic_cost = error.T @ Q @ error + dU.T @ R @ dU
    quadratic_problem = {'x': dU, 'f': quadratic_cost}

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
