import mpc
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


class MpcObj:
    # A = ca.DM([[1, 0], [0, 2]])
    # B = ca.DM([[0.1, 0], [0, 0.2]])
    # Hp = 40
    # Hu = 30
    #
    # Q = ca.DM([[1, 0], [0, 3]])
    # Qb = mpc.blockdiag(Q, Hp)
    #
    # R = ca.DM([[1, 0], [0, 2]])
    # Rb = mpc.blockdiag(R, Hu)
    # x0 = ca.DM([[10], [11]])
    # u_prev = ca.DM([-1, -3])
    # ref = ca.DM.ones(Hp * 2, 1)
    # ref[1::A.size1()] = np.cumsum(ref[1::A.size1()]) / 20 + 2

    def __init__(self, A, B, Hu, Hp, Q, R, x0, u_prev, ref, saved_steps=100):
        self.saved_steps = saved_steps
        self.A = A
        self.B = B
        self.Hp = Hp
        self.Hu = Hu
        self.Q = Q
        self.Qb = mpc.blockdiag(Q, Hp)
        self.R = R
        self.Rb = mpc.blockdiag(R, Hu)
        self.psi = mpc.gen_psi(A, Hp)
        self.upsilon = mpc.gen_upsilon(A, B, Hp)
        self.theta = mpc.gen_theta(self.upsilon, B, Hu)

        self.x0 = x0
        self.u_prev = u_prev
        self.ref = ref
        solver = mpc.gen_mpc_solver(A, B, Hu, Hp, self.Qb, self.Rb)
        self.solver_input = mpc.gen_solver_input(x0, u_prev, ref)
        self.result = solver(p=self.solver_input)
        self.dU = self.result['x']
        self.prev_steps = self.dU
       # self.prev_steps.append(self.result)
        self.predicted_states = mpc.gen_predicted_states(self.psi, self.x0, self.upsilon,
                                                         self.u_prev, self.theta, self.dU)

    def step(self, x0, u_prev, ref):
        self.solver_input = mpc.gen_solver_input(x0, u_prev, ref)
        self.result = self.solver(p=self.solver_input)

    def plot_step(self, options={}):
        # should plot
        #   Reference
        #   Predicted States
        #   Inputs
        #   (Costs: dU og State cost)

        opts = {'drawU': 'dU'}
        opts.update(options)
        states = self.A.size1()
        inputs = self.B.size2()

        t = np.arange(0, (self.predicted_states.size1() + 1) / 2)

        ax1 = plt.subplot(211)
        for s in range(0, states):
            col = plt.cm.tab10(s)

            plt.plot(t[1:], self.predicted_states[s::states], color=col)
            plt.plot(t[1:], self.ref[s::states], '--', color=col)
            plt.plot(t[0], self.x0[s], '.', color=col)

            plt.plot(t[0:2], ca.vertcat(self.x0[s], self.predicted_states[s]), 'r-')

        plt.subplot(212, sharex=ax1)

        t = np.append(np.arange(-1, 0), t)
        for i in range(0, inputs):
            col = plt.cm.tab10(i)
            w = 'post'

            if opts['drawU'] == 'U':

                U = np.cumsum(ca.vertcat(self.u_prev[i], self.dU[i::inputs]))

                plt.step(t[0:self.Hu + 1], U, color=col, where=w)
                #       plt.step(t[1:3], U[1:3], color=col, where=w)
                plt.plot(t[1], U[1], '.', color=col)

                plt.step(t[1:3], U[0:2], 'r-', where="pre")

            else:
                idU = self.dU[i::inputs]
                plt.plot(t[0:self.Hu], idU, '.', color=col)
                #plt.plot(t[0:2], ca.vertcat(self.u_prev[i], idU[0]), '.', color=col)
                plt.plot(t[1], idU[0], 'o', color=col)
                # plt.plot(t[0], idU[0], 'r.')

        plt.show()

    # def step(self):
    #
    #     x_prediction = mpc.gen_predicted_states(self.psi, self.x0, self.upsilon,
    #                                             self.u_prev, self.theta, )
    #     self.solver_input = mpc.gen_solver_input(x0, u_prev, ref)
    #     self.result = self.solver(p=self.solver_input)
