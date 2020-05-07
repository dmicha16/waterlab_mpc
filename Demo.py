import mpco
import casadi as ca
import mpc
import numpy as np
import matplotlib.pyplot as plt



# Real Model
A = ca.DM([[1, 0], [0, 1.5]])
B = ca.DM([[0.1, 0], [0, 0.5]])

states = A.size1()
inputs = B.size2()
# Predictive Model
rand_var = 0.1
const_var = 0.9
rand_A = np.random.rand(A.size1(), A.size2())

Ap = A * const_var + (np.random.rand(A.size1(), A.size2()) - 0.5) * rand_var
Bp = B * const_var + (np.random.rand(B.size1(), B.size2()) - 0.5) * rand_var
Hp = 40
Hu = 30

Q = ca.DM([[1, 0], [0, 3]])
Qb = mpc.blockdiag(Q, Hp)
R = ca.DM([[1, 0], [0, 2]])
Rb = mpc.blockdiag(R, Hu)

x0 = ca.DM([[10], [11]])
u0 = ca.DM([-1, -3])
ref = ca.DM.ones(Hp * 2, 1)
# ref[1::A.size1()] = np.cumsum(ref[1::A.size1()]) / 20 + 2


mmpc = mpco.MpcObj(Ap, Bp, Hu, Hp, Q, R, x0, u0, ref)
mmpc.plot_step({'drawU': 'U'})

steps = 50

x = x0
u = u0
x_array = x
u_array = u
looper = True


for j in range(1, steps):

    if looper:
        loop_in = input("press any key to step, or \'r\' to run all steps")
        looper = 'r' != loop_in

    u = u + mmpc.get_next_control_input_change()
    u_array = ca.horzcat(u_array, u)
    x = A @ x + B @ u
    x_array = ca.horzcat(x_array, ca.vec(x))
    mmpc.step(x, u, ref)
    mmpc.plot_step({'drawU': 'U'})
    mmpc.plot_progress({'drawU': 'U'})
        # draw




# draw
t = range(0, j + 1)

ax1 = plt.subplot(211)
for s in range(0, states):
    col = plt.cm.tab10(s)
    start = 0

    plt.plot(range(0, mmpc.get_k())[start:], mmpc.get_x0_log()[s, start:].T, color=col)
    plt.plot(range(0, mmpc.get_k())[start:], mmpc.get_expected_x_log()[s, start:].T, '--', color=col)

plt.subplot(212, sharex=ax1)

t = np.append(np.arange(-1, 0), t)
for i in range(0, inputs):
    col = plt.cm.tab10(i)
    plt.plot(range(0, mmpc.get_k() ), mmpc.get_dU_log()[i, :].T)

plt.show()
