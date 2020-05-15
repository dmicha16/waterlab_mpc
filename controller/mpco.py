from controller import mpc
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


class MpcObj:

    def __init__(self, A, B, Hu, Hp, Q, R, x0, u_prev, ref, B_d=None, disturbance=None, log_length=0, k=1):
        """
        :param A:(mxm) Model dynamics matrix of type casadi.DM
        :param B:(mxn) Input dynamics matrix of type casadi.DM
        :param Hu:(int) Control horizon of type Integer
        :param Hp: (int) Prediction horizon of type Integer
        :param Q:(mxm) State cost matrix of type casadi.DM
        :param R:(mxm) Input change cost matrix of type casadi.DM
        :param x0:(mx1) Initial states
        :param u_prev:(nx1) Initial contol input
        :param ref: (m*hU x 1) Reference trajectory
        :param log_length: (int) Nr. of steps to log, 0 = all
        :param k: (int) Initial time step
        """

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
        if B_d is None:
            self.B_d = B
        else:
            self.B_d = B_d
        self.theta_d = mpc.gen_theta(self.upsilon, self.B_d, Hp)
        self.solver = mpc.gen_mpc_solver(A, B, Hu, Hp, self.Qb, self.Rb, self.B_d)

        self.k = k
        self.x0 = x0
        self.u_prev = u_prev
        self.ref = ref
        if disturbance is None:
            self.disturbance = ca.DM.zeros(self.Hp * self.inputs, 1)
        else:
            self.disturbance = disturbance
        self.solver_input = mpc.gen_solver_input(self.x0, self.u_prev, self.ref, self.disturbance)
        lb = ca.cumsum(ca.DM.ones(self.states * self.Hp, 1)) * 0.1
        self.result = self.solver(p=self.solver_input)
        self.dU = self.result['x']
        self.prev_steps = self.dU
        # self.prev_steps.append(self.result)
        self.predicted_states = mpc.gen_predicted_states(self.psi, self.x0, self.upsilon, self.u_prev, self.theta,
                                                         self.dU, self.theta_d, self.disturbance)

        self.log_length = log_length
        self.log_x0 = ca.vec(x0)
        self.log_ref = ca.vec(self.get_next_ref())
        self.log_disturbance = ca.vec(self.get_next_disturbance())
        self.log_expected_x = ca.vec(self.get_next_expected_state())
        self.log_dU = ca.horzcat(ca.vec(u_prev), ca.vec(self.get_next_control_input_change()))

    def get_next_ref(self):
        return self.ref[:self.states]

    def get_next_disturbance(self):
        return self.disturbance[:self.inputs]

    def get_next_control_input_change(self):
        return self.dU[:self.inputs]

    def get_next_expected_state(self):
        return self.predicted_states[:self.states]

    def set_disturbance(self, disturbance):
        self.disturbance = disturbance

    def get_disturbance(self):
        return self.disturbance

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

    def set_B_d(self, B_d):
        self.lift(B_d=B_d)

    def get_B_d(self):
        return self.B_d

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

    def get_du(self):

        return self.dU

    def log(self, x0, expected_x, dU, ref, disturbance):

        self.log_x0 = ca.horzcat(self.log_x0, ca.vec(x0))
        self.log_ref = ca.horzcat(self.log_ref, ca.vec(ref))
        self.log_disturbance = ca.horzcat(self.log_disturbance, ca.vec(disturbance))
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

    def lift(self, A=None, B=None, Hu=None, Hp=None, Q=None, R=None, B_d=None):

        if A != None and A.shape == self.A.shape:
            self.A = A
        if B != None and B.shape == self.B.shape:
            self.B = B
        if B_d != None and B_d.shape == self.B_d.shape:
            self.B_d = B_d
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
        self.theta_d = mpc.gen_theta(self.upsilon, self.B_d, Hu)
        self.solver = mpc.gen_mpc_solver(self.A, self.B, self.Hu, self.Hp, self.Qb, self.Rb, self.B_d)

    def print_result(self):
        print(self.result)

    def print_solver(self):
        print(self.solver)

    def step(self, x0, u_prev, ref, disturbance=None):
        self.x0 = x0
        self.u_prev = u_prev
        self.ref = ref
        if disturbance is None:
            self.disturbance = ca.DM.zeros(self.Hp * self.inputs, 1)
        else:
            self.disturbance = disturbance
        self.solver_input = mpc.gen_solver_input(self.x0, self.u_prev, self.ref, self.disturbance)
        self.result = self.solver(p=self.solver_input)

        self.dU = self.result['x']
        self.prev_steps = self.dU
        # self.prev_steps.append(self.result)
        self.predicted_states = mpc.gen_predicted_states(self.psi, self.x0, self.upsilon, self.u_prev, self.theta,
                                                         self.dU, self.theta_d, self.disturbance)
        self.log(x0, self.get_next_expected_state(), self.get_next_control_input_change(), self.get_next_ref(),
                 self.get_next_disturbance())
        self.k = self.k + 1

    def plot_step(self, options={}, plot_pause=0.3):
        # should plot
        #   Reference
        #   Predicted States
        #   Inputs
        #   (Costs: dU og State cost)

        plt.clf()

        opts = {'drawU': 'U'}
        opts.update(options)

        t = np.arange(0, self.get_Hp() + 1)

        ax1 = plt.subplot(211)
        plt.title("Optimized step at time k = {}".format(self.k))
        for s in range(0, self.states):
            col = plt.cm.tab10(s)

            plt.plot(t[1:], self.predicted_states[s::self.states], color=col)
            plt.plot(t[1:], self.ref[s::self.states], '--', color=col)
            plt.plot(t[0:2], ca.vertcat(self.x0[s], self.predicted_states[s]), 'r-')
            plt.plot(t[0], self.x0[s], '.', color=col, label="State {}={:.3f}".format(s, float(self.x0[s])))

        plt.ylabel('States')
        ax1.legend()
        ax2 = plt.subplot(212, sharex=ax1)

        t = np.append(np.arange(-1, 0), t)
        for i in range(0, self.inputs):
            col = plt.cm.tab10(i)
            w = 'post'

            if opts['drawU'] == 'both' or opts['drawU'] == 'U':
                U = np.cumsum(ca.vertcat(self.u_prev[i], self.dU[i::self.inputs]))

                plt.step(t[0:self.Hu + 1], U, color=col, where=w)
                #       plt.step(t[1:3], U[1:3], color=col, where=w)
                plt.plot(t[1], U[1], '.', color=col, label="u{}={:.3f}".format(i, float(U[1])))

                plt.step(t[1:3], U[0:2], 'r-', where="pre")

            if opts['drawU'] == 'both' or opts['drawU'] == 'dU':
                idU = self.dU[i::self.inputs]
                plt.plot(t[1:self.Hu + 1], idU, '.', color=col)
                # plt.plot(t[0:2], ca.vertcat(self.u_prev[i], idU[0]), '.', color=col)
                plt.plot(t[1], idU[0], 'o', color=col, label="du{}={:.3f}".format(i, float(idU[1])))
                # plt.plot(t[0], idU[0], 'r.')

        ax2.legend()

        plt.xlabel('Time in steps')
        plt.ylabel(opts['drawU'])
        plt.draw()
        plt.pause(plot_pause)
        plt.show()

    def plot_progress(self, options={}, ignore_states=[], ignore_inputs=[], prev_n=20, plot_pause=0.3):
        # should plot
        #   Reference
        #   Predicted States
        #   Inputs
        #   (Costs: dU og State cost)

        plt.clf()

        opts = {'drawU': 'U'}
        opts.update(options)

        t = np.arange(-self.k, self.get_Hp() + 1)

        ax1 = plt.subplot(211)
        plt.title("Progress at time k = {}".format(self.k))
        plt.xlim([-self.k, self.Hp])

        for s in range(0, self.states):
            if s not in ignore_states:
                col = plt.cm.tab10(s)
                plt.plot(t[1:self.k + 1], self.get_x0_log()[s, 0:].T, color=col)
                plt.plot(t[2:self.k + 2], self.get_expected_x_log()[s, 0:].T, color=col, ls='-.')

                plt.plot(t[self.k + 1:], self.predicted_states[s::self.states], color=col, alpha=0.6)
                plt.plot(t[self.k + 1:], self.ref[s::self.states], '--', color=col, alpha=0.6)
                plt.plot(t[2:self.k + 2], self.get_ref_log()[s, :].T, '--', color=col)
                plt.plot(t[self.k:self.k + 2], ca.vertcat(self.x0[s], self.predicted_states[s]), 'r-')
                plt.plot(t[self.k], self.x0[s], '.', color=col, label="State {}={:.3f}".format(s, float(self.x0[s])))

        plt.ylabel('States')
        ax1.legend()
        ax2 = plt.subplot(212, sharex=ax1)

        for i in range(0, self.inputs):
            col = plt.cm.tab10(i)
            w = 'post'
            if i not in ignore_inputs:
                if opts['drawU'] == 'both' or opts['drawU'] == 'U':
                    # plt.plot(t[1:self.k + 1], self.get_x0_log()[s, 0:].T, color=col)

                    U = np.cumsum(ca.vertcat(self.get_dU_log()[i, 0:-1].T, self.dU[i::self.inputs]))

                    plt.step(t[0:self.k + self.Hu], U, color=col, where=w)
                    #       plt.step(t[1:3], U[1:3], color=col, where=w)
                    plt.plot(t[self.k], U[self.k], '.', color=col, label="U{}={:.3f}".format(i, float(U[self.k])))

                    plt.step(t[self.k:self.k + 2], U[self.k - 1:self.k + 1], 'r-', where="pre")

                if opts['drawU'] == 'both' or opts['drawU'] == 'dU':
                    idU = ca.vertcat(self.get_dU_log()[i, 0:-1].T, self.dU[i::self.inputs])

                    plt.plot(t[0:self.k + self.Hu], idU, '.', color=col)
                    # plt.plot(t[0:2], ca.vertcat(self.u_prev[i], idU[0]), '.', color=col)
                    plt.plot(t[self.k], idU[self.k], 'o', color=col, label="dU{}={:.3f}".format(i, float(idU[self.k])))
                    # plt.plot(t[0], idU[0], 'r.')
        ax2.legend()

        plt.xlabel('Time in steps')
        plt.ylabel(opts['drawU'])
        plt.draw()
        plt.pause(plot_pause)
        plt.show()

    # def step(self):
    #
    #     x_prediction = mpc.gen_predicted_states(self.psi, self.x0, self.upsilon,
    #                                             self.u_prev, self.theta, )
    #     self.solver_input = mpc.gen_solver_input(x0, u_prev, ref)
    #     self.result = self.solver(p=self.solver_input)
