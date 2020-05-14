from controller import mpc
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np


def plot_mpc_step(A, B, Hp, Hu, sol, p, options={}):
    # should plot
    #   Reference
    #   Predicted States
    #   Inputs
    #   (Costs: dU og State cost)

    opts = {'drawU': 'dU'}
    opts.update(options)
    states = A.size1()
    inputs = B.size2()

    x0 = p[0:states]
    u_prev = p[states:(states + inputs)]
    ref = p[(states + inputs):]

    dU = sol['x']

    psi = mpc.gen_psi(A, Hp)
    upsilon = mpc.gen_upsilon(A, B, Hp)
    theta = mpc.gen_theta(upsilon, B, Hu)
    predicted_states = mpc.gen_predicted_states(psi, x0, upsilon, u_prev, theta, dU)

    t = np.arange(0, (predicted_states.size1() + 1) / 2)
    t_prev = np.arange(-1, 1)

    ax1 = plt.subplot(211)
    for s in range(0, states):
        col = plt.cm.tab10(s)

        plt.plot(t[1:], predicted_states[s::states], color=col)
        plt.plot(t[1:], ref[s::states], '--', color=col)
        plt.plot(t[0], x0[s], '.', color=col)

        plt.plot(t[0:2], ca.vertcat(x0[s], predicted_states[s]), 'r-')

    plt.subplot(212, sharex=ax1)

    t = np.append(np.arange(-1, 0), t)
    for i in range(0, inputs):
        col = plt.cm.tab10(i)
        w = 'post'

        if opts['drawU'] == 'U':

            U = np.cumsum(ca.vertcat(u_prev[i], dU[i::inputs]))

            plt.step(t[0:Hu + 1], U, color=col, where=w)
            #       plt.step(t[1:3], U[1:3], color=col, where=w)
            plt.plot(t[1], U[1], '.', color=col)

            plt.step(t[1:3], U[0:2], 'r-', where="pre")

        else:
            idU = dU[i::inputs]
            plt.plot(t[0:Hu], idU, '.', color=col)
            plt.plot(t_prev[-2:], ca.vertcat(u_prev[i], idU[0]), '.', color=col)
            plt.plot(t[0], idU[0], 'o', color=col)
            # plt.plot(t[0], idU[0], 'r.')

    print(sol)
    print(predicted_states[0::2])
    print(predicted_states[1::2])
    plt.show()
    print(ref.size())
