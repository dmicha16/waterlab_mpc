import casadi as ca
from controller import mpc, mpco
import numpy as np
import matplotlib.pyplot as plt

# import matplotlib
# matplotlib.use('Qt4Agg',warn=False, force=True)
# from matplotlib import pyplot as plt


plt.ion()

# Prediction Model
Ap = ca.DM([[1, 0, 0], [0, 0.5, 0], [0, 0, 0.7]])
Bp = ca.DM([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 1]])
Bp_d = ca.DM([[0.1, 0, 0], [0, 0.5, 0], [0, 0, 1]])

states = Ap.size1()
inputs = Bp.size2()

# Real Model
rand_var = 0.00
const_var = 1.00
rand_A = np.random.rand(Ap.size1(), Ap.size2())

A = Ap * const_var + (np.random.rand(Ap.size1(), Ap.size2()) - 0.5) * rand_var
B = Bp * const_var + (np.random.rand(Bp.size1(), Bp.size2()) - 0.5) * rand_var
B_d = Bp_d * const_var + (np.random.rand(Bp_d.size1(), Bp_d.size2()) - 0.5) * rand_var
Hp = 40
Hu = 40

dist_magnitude = 5
dist = (np.random.rand(Hp * inputs) - 0.5) * dist_magnitude

Q = ca.DM([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
Qb = mpc.blockdiag(Q, Hp)
R = ca.DM([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
Rb = mpc.blockdiag(R, Hu)

x0 = ca.DM([[10], [11], [12]])
u0 = ca.DM([-1, -3, -6])

ref = ca.DM.ones(Hp * states, 1)
for state in range(states):
    ref[state::states] = ref[state::states] + state -2

mmpc = mpco.MpcObj(Ap, Bp, Hu, Hp, Q, R, x0, u0, ref, Bp_d, dist)
mmpc.plot_progress({'drawU': 'U'})
mmpc.print_solver()
mmpc.print_result()
steps = 50

x = x0
u = u0
x_array = x
u_array = u
looper = True
step_size = 3

for j in range(1, steps):



    u = u + mmpc.get_next_control_input_change()
    x = A @ x + B @ u + B_d @ dist[:inputs]
    nd = (np.random.rand(inputs) - 0.5) * dist_magnitude / 2
    dist = np.append(dist[inputs:], nd)

    if looper and j % step_size == 0:
        loop_in = input("press any key to step, or \'r\' to run all steps")
        looper = 'r' != loop_in
        mmpc.plot_progress(options={'drawU': 'U'}, ignore_inputs=[1, 2])
        # mmpc.plot_step({'drawU': 'both'})


    mmpc.step(x, u, ref, dist)

