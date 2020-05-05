import mpco
import casadi as ca
import mpc
import numpy as np

A = ca.DM([[1, 0], [0, 1.5]])
B = ca.DM([[0.1, 0], [0, 0.5]])
Hp = 40
Hu = 30

Q = ca.DM([[1, 0], [0, 3]])
Qb = mpc.blockdiag(Q, Hp)

R = ca.DM([[1, 0], [0, 2]])
Rb = mpc.blockdiag(R, Hu)
x0 = ca.DM([[10], [11]])
u_prev = ca.DM([-1, -3])
ref = ca.DM.ones(Hp * 2, 1)
ref[1::A.size1()] = np.cumsum(ref[1::A.size1()]) / 20 + 2

mmpc = mpco.MpcObj(A, B, Hu, Hp, Q, R, x0, u_prev, ref)

for i in range(0,10):
    input("press key")

