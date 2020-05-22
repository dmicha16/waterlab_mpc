import casadi as ca


def gen_mpc_solver(A, B, Hu, Hp, Q, R, B_d=None, S=None):
    '''
    Consult at p55 in Jan M. book for better understanding as well as casadi documentation
        :param A:(mxm) Model dynamics matrix of type casadi.DM
        :param B:(mxn) Input dynamics matrix of type casadi.DM
        :param Hu:(int) Control horizon of type Integer
        :param Hp: (int) Prediction horizon of type Integer
        :param Q:(mxm) State cost matrix of type casadi.DM
        :param R:(mxm) Input change cost matrix of type casadi.DM
        :param B_d:(mxn)
    '''

    if B_d is None:
        B_d = B
    # TODO: Fix cost on Inputs u - Weight matrix S
    if S is None:
        S = ca.DM.zeros(R.size1(), R.size2())
        S = R

    # Useful dimensions
    number_of_states = A.size1()
    number_of_inputs = B.size2()
    number_of_disturbances = B_d.size2()

    # Index for input variable
    initial_state_index_end = number_of_states
    prev_control_input_index_end = initial_state_index_end + number_of_inputs
    reference_index_end = prev_control_input_index_end + Hp * number_of_states
    prev_disturbance_index_end = reference_index_end + number_of_disturbances
    delta_disturbances_input_index_end = prev_disturbance_index_end + Hp * number_of_disturbances

    # Declaring and parting out input variables
    # Input = [x0, u_prev, ref, ud_prev, ud]
    input_variables = ca.SX.sym('i', delta_disturbances_input_index_end, 1)
    x0 = input_variables[0:initial_state_index_end, :]
    u_prev = input_variables[initial_state_index_end:prev_control_input_index_end, :]
    ref = input_variables[prev_control_input_index_end:reference_index_end, :]
    ud_prev = input_variables[reference_index_end:prev_disturbance_index_end, :]
    dud = input_variables[prev_disturbance_index_end:delta_disturbances_input_index_end, :]

    # Declaring solver outputs
    x = ca.SX.sym('x', number_of_inputs * Hu, 1)
    du = x[:number_of_inputs * Hu]

    # The wrong way of declaring inputs
    # x0 = ca.SX.sym('x0', A.shape[0], 1)
    # u_prev = ca.SX.sym('u_prev', B.shape[1], 1)
    # ref = ca.SX.sym('ref', Hp * A.shape[0], 1)
    # input_variables = ca.vertcat(ca.vertcat(x0, u_prev), ref)

    # To formulate a MPC optimization problem we need to describe:
    # predicted_states  = Z = psi x(k) + upsilon u(k-1) + Theta dU(x) + upsilon ud(k-1) + Theta dUd(x)
    psi = gen_psi(A, Hp)
    upsilon = gen_upsilon(A, B, Hp)
    theta = gen_theta(upsilon, B, Hu)
    upsilon_d = gen_upsilon(A, B_d, Hp)
    theta_d = gen_theta(upsilon_d, B_d, Hp)
    predicted_states = gen_predicted_states(psi, x0, upsilon, u_prev, theta, du, upsilon_d, ud_prev, theta_d, dud)

    # Setup constraints
    # construct U fom dU
    U = ca.SX.ones(du.size1())
    for i in range(0, number_of_inputs):
        U[i::number_of_inputs] = ca.cumsum(du[i::number_of_inputs])
    U = U + ca.repmat(u_prev, Hu, 1)
    constraints = ca.vertcat(predicted_states, U)

    # Cost function:
    # Cost = (Z - T)' * Q * (Z - T) + dU' * R * dU
    error = predicted_states - ref  # e = (Z - T)
    quadratic_cost = error.T @ Q @ error \
                     + du.T @ R @ du \
                     + U.T @ S @ U
    # Setup Solver
    # set print level: search for 'printLevel' in link
    # http://casadi.sourceforge.net/v3.1.0/api/internal/de/d94/qpoases__interface_8cpp_source.html
    opts = dict(printLevel='debug_iter')
    quadratic_problem = {'x': du, 'p': input_variables, 'f': quadratic_cost, 'g': constraints}
    mpc_solver = ca.qpsol('mpc_solver', 'qpoases', quadratic_problem, opts)
    # print(quadratic_cost)
    print(mpc_solver)

    return mpc_solver


def gen_psi(A, Hp):
    """
    Consult at p55 in Jan M. book for better understanding
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
    Consult at p55 in Jan M. book for better understanding
    :param A: Should be of type casadi.DM dimensions mxm
    :param B: Should be of type casadi.DM dimensions mxp
    :param Hp: Length of prediction horizon, should be an integer
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
    Consult at p55 in Jan M. book for better understanding
    :param upsilon: Should be of type casadi.DM
    :param B: Placeholder matrix Should be of type casadi.DM - Dimensions mxp
    :param Hu: Should be an integer
    :return: Of type casadi.DM
    """

    Theta = upsilon

    for col_idx in range(1, Hu):
        zeros_matrix = ca.DM.zeros(col_idx * B.shape[0], B.shape[1])

        # Reduce upsilon by the number of rows corresponding to the col_idx and rows of B
        reduced_upsilon = upsilon[0:(upsilon.shape[0] - B.shape[0] * col_idx):1, :]

        new_col = ca.vertcat(zeros_matrix, reduced_upsilon)

        Theta = ca.horzcat(Theta, new_col)

    return Theta


def gen_predicted_states(psi, x0, upsilon, u_prev, theta, du, upsilon_d=None, ud_prev=None, theta_d=None, dud=None):
    """
    Consult at p55 in Jan M. book for better understanding
    Note that the added disturbance is modeled as an input disturbance
    :param psi: Should be of type casadi.DM 
    :param x0: States at time x(k)- Of Type casadi.SX
    :param upsilon: Should be of type casadi.DM 
    :param u_prev: Inputs at time u(k-1) - Of Type casadi.SX
    :param theta: Should be of type casadi.DM 
    :param dU: Change in inputs from time u(k-1) - Of Type casadi.SX
    :return: Predicted states - Of Type casadi.SX
    """
    if upsilon_d is None:
        upsilon_d = upsilon
        print("Since no upsilon_d has been entered: upsilon_d = upsilon")
    if theta_d is None:
        theta_d = theta
        print("Since no theta_d has been entered: theta_d = theta")
    if dud is None:
        dud = ca.DM.zeros(theta_d.size2(), 1)
        print("Since no dud has been entered: dud = 0")
    if ud_prev is None:
        ud_prev = ca.DM.zeros(upsilon_d.size2(), 1)
        print("Since no dud has been entered: dud = 0")
    # Hp = 6
    # o = ca.vec(ca.DM([0, -2.535915115, 0, 0, 0, -20.48732092, 23.02323604]))
    # op = o
    # for i in range(0, Hp):
    #     op = ca.vertcat(op, o)

    x = psi @ x0 + \
        upsilon @ u_prev + \
        theta @ du + \
        upsilon_d @ ud_prev + \
        theta_d @ dud\
        # + op
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


def gen_solver_input(x0, u_prev, ref, ud_prev, disturbance):
    # TODO: Allow disturbance to be None?
    return ca.vertcat(ca.vec(x0),
                      ca.vertcat(ca.vec(u_prev),
                                 ca.vertcat(ca.vec(ref),
                                            ca.vertcat(ca.vec(ud_prev),
                                                       ca.vec(disturbance)))))
