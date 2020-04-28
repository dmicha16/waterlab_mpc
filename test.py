import mpc
import networkcontrol as co
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import mpc_plotter as mpl

A = ca.DM([[1, 0], [0, 1.5]])
B = ca.DM([[0.1, 0], [0, 0.11]])
Hp = 40
Hu = 30
I = ca.DM.eye(A.size1())
I = ca.DM([[1, 0], [0, 3]])
Q = mpc.blockdiag(I, Hp)
I = ca.DM.eye(B.size2())
I = ca.DM([[1, 0], [0, 2]])
R = mpc.blockdiag(I,Hu)
x0 = ca.DM([[10],[11]])
u_prev = ca.DM([0,0])
ref = ca.DM.ones(Hp*2, 1)

input_solver = mpc.gen_solver_input(x0, u_prev, ref)

mpc_solver = mpc.gen_mpc_solver(A, B, Hu, Hp, Q, R)

r = mpc_solver(p=input_solver)
psi = mpc.gen_psi(A, Hp)
upsilon = mpc.gen_upsilon(A, B, Hp)
theta = mpc.gen_theta(upsilon, B, Hu)



mpl.plot_mpc_step(A,B,Hp,Hu,r,input_solver)



# U = ca.SX(B.shape[1], Hp)
# for t in range(0, Hp):
#     if t < Hu:
#         U[:, t] = ca.sum2(dU[:, 0:t + 1]) + u_prev
#     else:
#         U[:, t] = U[:, t - 1]
