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
Bp_d = ca.DM([[1], [2], [0.5]])

states = Ap.size1()
inputs = Bp.size2()
disturbances = Bp_d.size2()

# Real Model
rand_var = 0.00
const_var = 1.00
rand_A = np.random.rand(Ap.size1(), Ap.size2())

A = Ap * const_var + (np.random.rand(Ap.size1(), Ap.size2()) - 0.5) * rand_var
B = Bp * const_var + (np.random.rand(Bp.size1(), Bp.size2()) - 0.5) * rand_var
B_d = Bp_d * const_var + (np.random.rand(Bp_d.size1(), Bp_d.size2()) - 0.5) * rand_var
Hp = 40
Hu = 40

dist_magnitude = 0.5
dist = ca.DM((np.random.rand(Hp * disturbances) - 0.5) * dist_magnitude)
initial_disturbance = ca.DM.zeros(disturbances)

Q = ca.DM([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
Qb = mpc.blockdiag(Q, Hp)
R = ca.DM([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
Rb = mpc.blockdiag(R, Hu)

x0 = ca.DM([[10], [11], [12]])
u0 = ca.DM([-1, -3, -6])

ref = ca.DM.ones(Hp * states, 1)
for state in range(states):
    ref[state::states] = ref[state::states] + state - 2

lower_bounds_input = ca.DM.ones(inputs * Hu) * -15
lower_bounds_slew_rate = ca.DM.ones(inputs * Hu) * -10
lower_bounds_states = ca.DM.ones(states * Hp) * -1
upper_bounds_input = ca.DM.ones(inputs * Hu) * 11
upper_bounds_slew_rate = ca.DM.ones(inputs * Hu) * 10
upper_bounds_states = ca.DM.ones(states * Hp) * 10
mmpc = mpco.MpcObj(Ap, Bp, Hu, Hp, Q, R,  ref=ref, initial_control_signal=u0, input_matrix_d=Bp_d,
                   lower_bounds_input=lower_bounds_input,
                   lower_bounds_slew_rate=lower_bounds_slew_rate, upper_bounds_slew_rate=upper_bounds_slew_rate,
                   upper_bounds_input=upper_bounds_input, lower_bounds_states=lower_bounds_states,
                   upper_bounds_states=upper_bounds_states)
mmpc.step(x0,u0,ref,initial_disturbance,dist)

mmpc.plot_progress({'drawU': 'U'})
mmpc.print_solver()
mmpc.print_result()
steps = 100

x = x0
u = u0
x_array = x

u_array = u
looper = True
step_size = 3
cum_dist = ca.DM(initial_disturbance)
for j in range(1, steps):

    # this model is instead of the EPA swmm
    u = u + mmpc.get_next_control_input_change()
    cum_dist = cum_dist + mmpc.get_next_disturbance_change()
    x = A @ x + B @ u + B_d @ cum_dist
    nd = ca.DM((np.random.rand(disturbances) - 0.5) * dist_magnitude / 2)

    dist = ca.vertcat(dist[disturbances:], nd)

    if looper and j % step_size == 0:
        loop_in = input("press any key to step, or \'r\' to run all steps")
        looper = ('r' != loop_in)
        mmpc.plot_progress(options={'drawU': 'U'}, ignore_inputs=[1])
        # mmpc.plot_step({'drawU': 'both'})

    mmpc.step(x, u, ref, prev_disturbance=cum_dist, disturbance=dist)
