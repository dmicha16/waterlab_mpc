import casadi as ca


# To formulate a MPC optimization problem we need to describe:
# Z = psi x(k) + upsilon u(k-1) + Theta dU(x) (Assuming no disturbance)
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
    :param upsilon: Should be of type casadi.DM - Dimensions mxm
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