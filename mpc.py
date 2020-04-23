import casadi as ca


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

def gen_theta(A, B, Hp, Hu):
    """
    :param A: Should be of type casadi.DM dimensions mxm
    :param B: Should be of type casadi.DM dimensions mxp
    :param Hp: Should be an integer
    :param Hu: Should be an integer
    :return: Of type casadi.DM
    """
    upsilon = gen_upsilon(A, B, Hp)
    Theta = upsilon

    for i in range(1, Hu):
        newcol = ca.vertcat(ca.DM.zeros(i*B.shape[0], B.shape[1]),
                            upsilon[0:(upsilon.shape[0] - B.shape[0] * i):1, :])
        Theta = ca.horzcat(Theta, newcol)

    return Theta

def gen_theta(upsilon, B, Hu):
    """
    :param upsilon: Should be of type casadi.DM
    :param B: Should be of type casadi.DM - Dimensions mxp
    :param Hu: Should be an integer
    :return: Of type casadi.DM
    """
    Theta = upsilon

    for i in range(1, Hu):
        newcol = ca.vertcat(ca.DM.zeros(i*B.shape[0], B.shape[1]),
                            upsilon[0:(upsilon.shape[0] - B.shape[0] * i):1, :])
        Theta = ca.horzcat(Theta, newcol)
    return Theta

def gen_predicted_states(psi, x0, upsilon, u_prev, theta, dU):
    """
    :param psi: Should be of type casadi.DM 
    :param x0: States at time x(k)- Of Type casadi.SX
    :param upsilon: Should be of type casadi.DM 
    :param u_prev: Inputs at time u(k-1) - Of Type casadi.SX
    :param theta: Should be of type casadi.DM 
    :param dU: Change in inputs from time u(k-1) - Of Type casadi.SX
    :return: Predicted states - Of Type casadi.SX
    """
    x = psi @ x0 + upsilon @ u_prev + theta @ dU
    return x

def blockdiag(Q,Hp):
    """
    :param Q: A mxm matrix - Of type casadi.DM
    :param Hp: Number of diagonal entries - Integer
    :return: (m * Hp) X (m * Hp) block diagonal matrix - Of type casadi.DM
    """
    R = ca.vertcat(Q, ca.DM.zeros((Hp-1) * Q.shape[0], Q.shape[1]))

    for i in range(1, Hp):
        top_row = ca.DM.zeros(i * Q.shape[0], Q.shape[1])
        bot_row = ca.DM.zeros((Hp-i-1) * Q.shape[0], Q.shape[1])
        new_row = ca.vertcat(ca.vertcat(top_row, Q), bot_row)

        R = ca.horzcat(R, new_row)

    return R


