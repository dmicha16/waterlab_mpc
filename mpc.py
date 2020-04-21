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
    for i in range(2, Hp):
        psi = ca.vertcat(psi, ca.mpower(A, i))

    return psi


def gen_upsilon(A, B, Hp):
    """
    :param A: Should be of type casadi.DM dimensions mxm
    :param B: Should be of type casadi.DM dimensions mxp
    :param Hp: Hp: Should be an integer
    :return: Of type casadi.DM
    """
    upsilon = B
    prev = upsilon
    for i in range(1, Hp - 1):
        new = ca.mpower(A, i) @ B + prev
        prev = new
        upsilon = ca.vertcat(upsilon, new)

    return upsilon
