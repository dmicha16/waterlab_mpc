import casadi as ca


def gen_mpc_solver(A, B, Hu, Hp, Q, R, B_d=None, L=None, Oc=None):
    # TODO: should Q and R be generated in function as other matricies?

    # # The wrong way of declaring inputs
    # Solver inputs
    # x0 = ca.SX.sym('x0', A.shape[0], 1)
    # u_prev = ca.SX.sym('u_prev', B.shape[1], 1)
    # ref = ca.SX.sym('ref', Hp * A.shape[0], 1)
    # input_variables = ca.vertcat(ca.vertcat(x0, u_prev), ref)

    states = A.size1()
    inputs = B.size2()
    if Oc is None:
        overflows = 0
    else:
        overflows = Oc.size2()

    # Solver inputs
    input_variables = ca.SX.sym('i', states + inputs + Hp * states + Hp * inputs, 1)
    x0 = input_variables[0:states, :]
    u_prev = input_variables[x0.size1():x0.size1() + inputs, :]
    ref = input_variables[x0.size1() + u_prev.size1():x0.size1() + u_prev.size1() + Hp * states, :]
    disturbance = input_variables[
                  x0.size1() + u_prev.size1() + ref.size1():x0.size1() + u_prev.size1() + ref.size1() + Hp * inputs, :]
    # Solver outputs
    x = ca.SX.sym('x', inputs * Hu + overflows * Hp, 1)
    dU = x[:inputs * Hu]
    dO = x[-overflows * Hp:]
    # To formulate a MPC optimization problem we need to describe:
    # Z = psi x(k) + upsilon u(k-1) + Theta dU(x) (Assuming no disturbance)
    psi = gen_psi(A, Hp)
    upsilon = gen_upsilon(A, B, Hp)
    theta = gen_theta(upsilon, B, Hu)
    if B_d is None:
        B_d = B
    theta_d = gen_theta(upsilon, B_d, Hp)
    predicted_states = gen_predicted_states(psi, x0, upsilon, u_prev, theta, dU, theta_d, disturbance)

    # Cost function:
    # Cost = (Z - T)' * Q * (Z - T) + dU' * R * dU
    error = predicted_states - ref  # e = (Z - T)
    constraints = predicted_states
    lb = ca.DM.zeros(constraints.size1(), 1)
    quadratic_cost = error.T @ Q @ error + dU.T @ R @ dU
    quadratic_problem = {'x': dU, 'p': input_variables, 'f': quadratic_cost, 'g': constraints}
    print(quadratic_cost)

    mpc_solver = ca.qpsol('mpc_solver', 'qpoases', quadratic_problem)
    print(mpc_solver)

    return mpc_solver


def gen_psi(A, Hp):
    """
    :param A: Should be of type casadi.DM
    :param Hp: Should be an intenger
    :return: Of type casadi.DM
    """
    psi = A
    for i in range(2, (Hp + 1)):
        psi = ca.vertcat(psi, ca.mpower(A, i))

    return psi


def gen_upsilon(A, B, Hp):
    """
    :param A: Should be of type casadi.DM dimensions mxm
    :param B: Should be of type casadi.DM dimensions mxp
    :param Hp: Should be an integer
    :return: Of type casadi.DM
    """
    upsilon = B
    prev = upsilon
    for i in range(1, Hp):
        new = ca.mpower(A, i) @ B + prev
        prev = new
        upsilon = ca.vertcat(upsilon, new)

    return upsilon


# def gen_theta(A, B, Hp, Hu):
#     """
#     :param A: Should be of type casadi.DM dimensions mxm
#     :param B: Should be of type casadi.DM dimensions mxp
#     :param Hp: Should be an integer
#     :param Hu: Should be an integer
#     :return: Of type casadi.DM
#     """
#     upsilon = gen_upsilon(A, B, Hp)
#     Theta = upsilon
#
#     for i in range(1, Hu):
#         newcol = ca.vertcat(ca.DM.zeros(i*B.shape[0], B.shape[1]),
#                             upsilon[0:(upsilon.shape[0] - B.shape[0] * i):1, :])
#         Theta = ca.horzcat(Theta, newcol)
#
#     return Theta

def gen_theta(upsilon, B, Hu):
    """
    :param upsilon: Should be of type casadi.DM
    :param B: Should be of type casadi.DM - Dimensions mxp
    :param Hu: Should be an integer
    :return: Of type casadi.DM
    """
    Theta = upsilon

    for i in range(1, Hu):
        newcol = ca.vertcat(ca.DM.zeros(i * B.shape[0], B.shape[1]),
                            upsilon[0:(upsilon.shape[0] - B.shape[0] * i):1, :])
        Theta = ca.horzcat(Theta, newcol)
    return Theta


def gen_predicted_states(psi, x0, upsilon, u_prev, theta, dU, theta_d=None, disturbance=None):
    """
    :param psi: Should be of type casadi.DM 
    :param x0: States at time x(k)- Of Type casadi.SX
    :param upsilon: Should be of type casadi.DM 
    :param u_prev: Inputs at time u(k-1) - Of Type casadi.SX
    :param theta: Should be of type casadi.DM 
    :param dU: Change in inputs from time u(k-1) - Of Type casadi.SX
    :return: Predicted states - Of Type casadi.SX
    """
    if disturbance is None:
        disturbance = ca.DM.zeros(psi.size1(), x0.size2())
    if theta_d is None:
        theta_d = theta
        disturbance = disturbance[:theta_d.size2(), :]
    x = psi @ x0 + \
        upsilon @ u_prev + \
        theta @ dU + \
        theta_d @ disturbance
    return x


def blockdiag(Q, Hp):
    """
    :param Q: A mxm matrix - Of type casadi.DM
    :param Hp: Number of diagonal entries - Integer
    :return: (m * Hp) X (m * Hp) block diagonal matrix - Of type casadi.DM
    """
    R = ca.vertcat(Q, ca.DM.zeros((Hp - 1) * Q.shape[0], Q.shape[1]))

    for i in range(1, Hp):
        top_row = ca.DM.zeros(i * Q.shape[0], Q.shape[1])
        bot_row = ca.DM.zeros((Hp - i - 1) * Q.shape[0], Q.shape[1])
        new_row = ca.vertcat(ca.vertcat(top_row, Q), bot_row)

        R = ca.horzcat(R, new_row)

    return R


def gen_solver_input(x0, u_prev, ref, disturbance=None):
    # TODO: Allow disturbance to be None?
    return ca.vertcat(ca.vec(x0), ca.vertcat(ca.vec(u_prev), ca.vertcat(ca.vec(ref), ca.vec(disturbance))))
