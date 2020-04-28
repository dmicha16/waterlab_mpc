import mpc
import networkcontrol as co
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

def plot_mpc_step(A,B,Hp,Hu,sol,p):
    # should plot
    #   Reference
    #   Predicted States
    #   Inputs
    #   (Costs: dU og State cost)

    states = A.size1()
    inputs= B.size2()

    x0 = p[0:states]
    u_prev = p[states:(states+inputs)]
    ref = p[(states+inputs):]

    dU=sol['x']

    psi = mpc.gen_psi(A, Hp)
    upsilon = mpc.gen_upsilon(A, B, Hp)
    theta = mpc.gen_theta(upsilon, B, Hu)
    predicted_states = mpc.gen_predicted_states(psi, x0, upsilon, u_prev, theta, dU)

    t = np.arange(0, (predicted_states.size1()+1)/2)
    t_prev = np.arange(-1, 1)

    ax1=plt.subplot(211)
    for s in range(0, states):
        col = plt.cm.tab10(s)

        plt.plot(t[1:], predicted_states[s::states], color=col)
        plt.plot(t[0], x0[s], '.', color=col)

        plt.plot(t[0:2], ca.vertcat(x0[s], predicted_states[s]), 'r-')

    plt.subplot(212,sharex=ax1)

    for i in range(0, inputs):
        col = plt.cm.tab10(i)
        w = 'post'
        plt.step(t[0:Hu], dU[i::inputs], color=col, where=w)
        plt.step(t_prev[-2:],  ca.vertcat(u_prev[i], dU[i]), color=col, where=w)
        plt.plot(t[0], dU[i], '.', color=col)

        plt.step(t[:2], dU[i:inputs*2:inputs], 'r-',where='post')


    print(sol)
    print(predicted_states[0::2])
    print(predicted_states[1::2])
    plt.show()
    print(ref.size())
