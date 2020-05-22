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

Ap = ca.DM([[1, 0, 0, 0, 0, 0, 0],
            [0, -0.3429575580, 1.280457558, 0, 0, 0, 0],
            [0, 1.342957558, -1.623415116, 1.280457558, 0, 0, 0],
            [0, 0, 1.342957558, -1.623415116, 1.280457558, 0, 0],
            [0, 0, 0, 1.342957558, -1.623415116, 1.280457558, 0],
            [0, 0, 0, 0, 1.342957558, -1.623415116, 1.2],
            [0, 0, 0, 0, 0, 1.3, -1.623415116]])

Ap = ca.DM([[1, 0, 0, 0, 0, 0, 0],
                [0, -0.3429575580, 1.280457558, 0, 0, 0, 0],
                [0, 1.342957558, -1.623415116, 1.280457558, 0, 0, 0],
                [0, 0, 1.342957558, -1.623415116, 1.280457558, 0, 0],
                [0, 0, 0, 1.342957558, -1.623415116, 1.280457558, 0],
                [0, 0, 0, 0, 1.342957558, -10.58661802, 10.24366046],
                [0, 0, 0, 0, 0, 10.30616046, -9.24366046]])

Bp = ca.DM([[-2 / 5, 0, -2 / 5, 0], [3 / 2, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
            [0, -12 / 5, 0, -12 / 5]])
Bp_d = ca.DM([2 / 5, 0, 0, 0, 0, 0, 0])

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
Hp = 8
Hu = Hp

dist_magnitude = 0.001
dist = ca.DM((np.random.rand(Hp * disturbances) +0) * dist_magnitude)
dist = ca.DM.ones(Hp * disturbances)*dist_magnitude
initial_disturbance = ca.DM.zeros(disturbances)

Q = ca.DM([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
Qb = mpc.blockdiag(Q, Hp)
R = ca.DM([[0.1, 0, 0], [0, 0.2, 0], [0, 0, 0.3]])
Rb = mpc.blockdiag(R, Hu)
Q = ca.DM(np.identity(7)) * 1
Q[0,0] = 10
R = ca.DM([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1000, 0], [0, 0, 0, 1000]])

x0 = ca.DM([[2], [0], [0], [0], [0], [0], [0]])
u0 = ca.DM([0, 0, 0, 0])

operating_point = ca.DM([0., -1.2305, 0., 0., 0., -3.6216, 4.8521])

ref = ca.DM.ones(Hp * states, 1)
for state in range(states):
    ref[state::states] = ref[state::states] + state * 0.1

lower_bounds_input = ca.DM.ones(inputs) * 0
lower_bounds_slew_rate = ca.DM.ones(inputs) * -10
lower_bounds_states = ca.DM.ones(states) * 0
upper_bounds_input = ca.DM.ones(inputs) * 111
upper_bounds_slew_rate = ca.DM.ones(inputs) * 100
upper_bounds_states = ca.DM.ones(states) * 1000
# lower_bounds_input = None
# lower_bounds_slew_rate = None
# lower_bounds_states = None
# upper_bounds_input = None
# upper_bounds_slew_rate = None
# upper_bounds_states = None
#lower_bounds_states = operating_point -100

mmpc = mpco.MpcObj(Ap, Bp, Hu, Hp, Q, R, ref=ref, initial_control_signal=u0, input_matrix_d=Bp_d,
                   lower_bounds_input=lower_bounds_input,
                   lower_bounds_slew_rate=lower_bounds_slew_rate, upper_bounds_slew_rate=upper_bounds_slew_rate,
                   upper_bounds_input=upper_bounds_input, lower_bounds_states=lower_bounds_states,
                   upper_bounds_states=upper_bounds_states,operating_point=operating_point)
mmpc.step(x0, u0, ref, initial_disturbance, dist)

mmpc.plot_progress({'drawU': 'U'})
mmpc.print_solver()
mmpc.print_result()
steps = 100

x = x0
u = u0
x_array = x

u_array = u
looper = True
step_size = 10
cum_dist = ca.DM(initial_disturbance)
for j in range(1, steps):

    # this model is instead of the EPA swmm
    u = u + mmpc.get_next_control_input_change()

    x = A @ x + B @ u + B_d @ cum_dist

    if looper and j % step_size == 0:
        loop_in = input("press any key to step, or \'r\' to run all steps")
        looper = ('r' != loop_in)
        mmpc.plot_progress(options={'drawU': 'U'})
        # mmpc.plot_step({'drawU': 'both'})

    mmpc.step(x, u, ref, prev_disturbance=cum_dist, disturbance=dist)
    nd = ca.DM((np.random.rand(disturbances) - 0.5) * dist_magnitude / 2)
    nd = ca.DM.ones(disturbances) * 0

    dist = ca.vertcat(dist[disturbances:], nd)
    cum_dist = cum_dist + mmpc.get_next_disturbance_change()
