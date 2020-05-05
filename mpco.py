import mpc
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


class MpcObj:

    def __init__(self, A, B, Hu, Hp, Q, R, x0, u_prev, ref, saved_steps=30, k=1):

        self.states = A.size1()
        self.inputs = B.size2()

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
        self.solver = mpc.gen_mpc_solver(A, B, Hu, Hp, self.Qb, self.Rb)

        self.k = k
        self.x0 = x0
        self.u_prev = u_prev
        self.ref = ref

        self.solver_input = mpc.gen_solver_input(x0, u_prev, ref)
        self.result = self.solver(p=self.solver_input)
        self.dU = self.result['x']
        self.prev_steps = self.dU
        # self.prev_steps.append(self.result)
        self.predicted_states = mpc.gen_predicted_states(self.psi, self.x0, self.upsilon, self.u_prev, self.theta,
                                                         self.dU)

        self.saved_steps = saved_steps
        self.log_x0 = ca.vec(x0)
        self.log_ref = ca.vec(self.get_next_ref())
        self.log_expected_x = ca.vec(self.get_next_expected_state())
        self.log_dU = ca.vec(self.get_next_control_input_change())

    def get_next_ref(self):
        return self.ref[:self.states]

    def get_next_control_input_change(self):
        return self.dU[:self.inputs]

    def get_next_expected_state(self):
        return self.predicted_states[:self.states]

    def get_k(self):
        return self.k

    def set_A(self, A):
        self.lift(A=A)

    def get_A(self):
        return self.A

    def set_B(self, B):
        self.lift(B=B)

    def get_B(self):
        return self.B

    def set_Hu(self, Hu):
        self.lift(Hu=Hu)

    def get_Hu(self):
        return self.Hu

    def set_Hp(self, Hp):
        self.lift(Hp=Hp)

    def get_Hp(self):
        return self.Hp

    def set_Q(self, Q):
        self.lift(Q=Q)

    def get_Q(self):
        return self.Q

    def set_R(self, R):
        self.lift(R=R)

    def get_R(self):
        return self.R

    def lift(self, A=None, B=None, Hu=None, Hp=None, Q=None, R=None):

        if A != None and A.shape == self.A.shape:
            self.A = A
        if B != None and B.shape == self.B.shape:
            self.B = B
        if Hp != None:
            self.Hp = Hp
        if Hu != None:
            self.Hu = Hu
        if Q != None and Q.shape == self.Q.shape:
            self.Q = Q
            self.Qb = mpc.blockdiag(Q, Hp)
        if R != None and R.shape == self.Q.shape:
            self.R = R
            self.Rb = mpc.blockdiag(R, Hu)

        self.psi = mpc.gen_psi(self.A, self.Hp)
        self.upsilon = mpc.gen_upsilon(self.A, self.B, self.Hp)
        self.theta = mpc.gen_theta(self.upsilon, self.B, self.Hu)
        self.solver = mpc.gen_mpc_solver(self.A, self.B, self.Hu, self.Hp, self.Qb, self.Rb)

    def get_du(self):

        return self.dU

    def log(self, x0, expected_x, dU, ref):

        self.log_x0 = ca.horzcat(self.log_x0, ca.vec(x0))
        self.log_ref = ca.horzcat(self.log_ref, ca.vec(ref))
        self.log_expected_x = ca.horzcat(self.log_expected_x, ca.vec(expected_x))
        self.log_dU = ca.horzcat(self.log_dU, ca.vec(dU))

    def get_expected_x_log(self):

        return self.log_expected_x

    def get_x0_log(self):

        return self.log_x0

    def get_ref_log(self):

        return self.log_ref

    def get_dU_log(self):

        return self.log_dU

    def step(self, x0, u_prev, ref):
        self.x0 = x0
        self.u_prev = u_prev
        self.ref = ref

        self.solver_input = mpc.gen_solver_input(x0, u_prev, ref)
        self.result = self.solver(p=self.solver_input)

        self.dU = self.result['x']
        self.prev_steps = self.dU
        # self.prev_steps.append(self.result)
        self.predicted_states = mpc.gen_predicted_states(self.psi, self.x0, self.upsilon,
                                                         self.u_prev, self.theta, self.dU)
        self.log(x0, self.get_next_expected_state(), self.get_next_control_input_change(), self.get_next_ref())
        self.k = self.k + 1

    def plot_step(self, options={}):
        # should plot
        #   Reference
        #   Predicted States
        #   Inputs
        #   (Costs: dU og State cost)

        opts = {'drawU': 'dU'}
        opts.update(options)

        t = np.arange(0, (self.predicted_states.size1() + 1) / 2)

        ax1 = plt.subplot(211)
        for s in range(0, self.states):
            col = plt.cm.tab10(s)

            plt.plot(t[1:], self.predicted_states[s::self.states], color=col)
            plt.plot(t[1:], self.ref[s::self.states], '--', color=col)
            plt.plot(t[0:2], ca.vertcat(self.x0[s], self.predicted_states[s]), 'r-')
            plt.plot(t[0], self.x0[s], '.', color=col, label=s)

        plt.ylabel('States')
        ax1.legend()
        ax2 = plt.subplot(212, sharex=ax1)

        t = np.append(np.arange(-1, 0), t)
        for i in range(0, self.inputs):
            col = plt.cm.tab10(i)
            w = 'post'

            if opts['drawU'] == 'U':

                U = np.cumsum(ca.vertcat(self.u_prev[i], self.dU[i::self.inputs]))

                plt.step(t[0:self.Hu + 1], U, color=col, where=w)
                #       plt.step(t[1:3], U[1:3], color=col, where=w)
                plt.plot(t[1], U[1], '.', color=col, label=i)

                plt.step(t[1:3], U[0:2], 'r-', where="pre")

            else:
                idU = self.dU[i::self.inputs]
                plt.plot(t[0:self.Hu], idU, '.', color=col)
                # plt.plot(t[0:2], ca.vertcat(self.u_prev[i], idU[0]), '.', color=col)
                plt.plot(t[1], idU[1], 'o', color=col, label="{}={}".format(i, idU[1]))
                # plt.plot(t[0], idU[0], 'r.')
        ax2.legend()
        plt.xlabel('Time in steps')
        plt.ylabel(opts['drawU'])
        plt.show()

    def plot_progress(self, options={}, ignore_states=[], ignore_inputs=[], prev_n=20):
        # should plot
        #   Reference
        #   Predicted States
        #   Inputs
        #   (Costs: dU og State cost)

        opts = {'drawU': 'dU'}
        opts.update(options)

        t = np.arange(-self.k, (self.predicted_states.size1() + 1) / 2)

        ax1 = plt.subplot(211)
        plt.xlim([-self.k, self.Hp])

        for s in range(0, self.states):
            if not ignore_states.__contains__(s):
                col = plt.cm.tab10(s)
                plt.plot(t[1:self.k + 1], self.get_x0_log()[s, 0:].T, color=col)
                plt.plot(t[2:self.k + 2], self.get_expected_x_log()[s, 0:].T, color=col, ls='--')

                plt.plot(t[self.k + 1:], self.predicted_states[s::self.states], color=col)
                plt.plot(t[self.k + 1:], self.ref[s::self.states], '--', color=col)
                plt.plot(t[self.k:self.k + 2], ca.vertcat(self.x0[s], self.predicted_states[s]), 'r-')
                plt.plot(t[self.k], self.x0[s], '.', color=col, label=s)

        plt.ylabel('States')
        ax1.legend()
        ax2 = plt.subplot(212, sharex=ax1)

        t = np.append(np.arange(-1, 0), t)
        for i in range(0, self.inputs):
            col = plt.cm.tab10(i)
            w = 'post'
            if not ignore_inputs.__contains__(s):
                if opts['drawU'] == 'U':

                    U = np.cumsum(ca.vertcat(self.u_prev[i], self.dU[i::self.inputs]))

                    plt.step(t[self.k:self.k + self.Hu + 1], U, color=col, where=w)
                    #       plt.step(t[1:3], U[1:3], color=col, where=w)
                    plt.plot(t[self.k + 1], U[1], '.', color=col, label=i)

                    plt.step(t[self.k + 1:self.k + 3], U[0:2], 'r-', where="pre")

                else:
                    idU = self.dU[i::self.inputs]
                    plt.plot(t[self.k + 0:self.k + self.Hu], idU, '.', color=col)
                    # plt.plot(t[0:2], ca.vertcat(self.u_prev[i], idU[0]), '.', color=col)
                    plt.plot(t[self.k + 1], idU[1], 'o', color=col, label="{}={}".format(i, idU[1]))
                    # plt.plot(t[0], idU[0], 'r.')
        ax2.legend()
        plt.xlabel('Time in steps')
        plt.ylabel(opts['drawU'])
        plt.show()

    # def step(self):
    #
    #     x_prediction = mpc.gen_predicted_states(self.psi, self.x0, self.upsilon,
    #                                             self.u_prev, self.theta, )
    #     self.solver_input = mpc.gen_solver_input(x0, u_prev, ref)
    #     self.result = self.solver(p=self.solver_input)
