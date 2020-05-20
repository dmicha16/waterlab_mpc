from controller import mpc
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


class MpcObj:

    def __init__(self, dynamics_matrix, input_matrix, Hu, Hp, Q, R, initial_state, u_prev, ref, input_matrix_d=None, disturbance=None, log_length=0, k=1,
                 lower_bounds_states=None, upper_bounds_states=None,
                 lower_bounds_slew_rate=None, upper_bounds_slew_rate=None,  # TODO slew
                 lower_bounds_input=None, upper_bounds_input=None):

        """
        :param dynamics_matrix:(mxm) Model dynamics matrix of type casadi.DM
        :param input_matrix:(mxn) Input dynamics matrix of type casadi.DM
        :param Hu:(int) Control horizon of type Integer
        :param Hp: (int) Prediction horizon of type Integer
        :param Q:(mxm) State cost matrix of type casadi.DM
        :param R:(mxm) Input change cost matrix of type casadi.DM
        :param initial_state:(mx1) Initial states
        :param u_prev:(nx1) Initial contol input
        :param ref: (m*hU x 1) Reference trajectory
        :param input_matrix_d:
        :param disturbance:
        :param log_length: (int) Nr. of steps to log, 0 = all TODO implement functionality
        :param k: (int) Initial time step TODO Check usage
        :param lower_bounds_states: (states*Hp x 1) Lower bounds vector for states as type casadi.DM
        :param upper_bounds_states: (states*Hp x 1) Upper bounds vector for states as type casadi.DM
        :param lower_bounds_slew_rate: (inputs*Hu x 1) Lower bounds vector for slew rates as type casadi.DM
        :param upper_bounds_slew_rate: (inputs*Hu x 1) Upper bounds vector for slew rates as type casadi.DM
        :param lower_bounds_input: (inputs*Hu x 1) Lower bounds vector for inputs as type casadi.DM
        :param upper_bounds_input: (inputs*Hu x 1) Lower bounds vector for slew rates as type casadi.DM
        """

        # Getting number of states and inputs from matrix dimensions
        self.states = dynamics_matrix.size1()
        self.inputs = input_matrix.size2()

        # Save dynamics and solver variables
        self.dynamics_matrix = dynamics_matrix
        self.input_matrix = input_matrix
        self.Hp = Hp
        self.Hu = Hu
        self.Q = Q
        self.Qb = mpc.blockdiag(Q, Hp)
        self.R = R
        self.Rb = mpc.blockdiag(R, Hu)
        self.psi = mpc.gen_psi(dynamics_matrix, Hp)
        self.upsilon = mpc.gen_upsilon(dynamics_matrix, input_matrix, Hp)
        self.theta = mpc.gen_theta(self.upsilon, input_matrix, Hu)
        if input_matrix_d is None:
            self.input_matrix_d = input_matrix
        else:
            self.input_matrix_d = input_matrix_d
        self.theta_d = mpc.gen_theta(self.upsilon, self.input_matrix_d, Hp)
        self.solver = mpc.gen_mpc_solver(dynamics_matrix, input_matrix, Hu, Hp, self.Qb, self.Rb, self.input_matrix_d)

        # initial step
        self.k = k
        self.initial_state = initial_state
        self.u_prev = u_prev
        self.ref = ref
        if disturbance is None:
            self.disturbance = ca.DM.zeros(self.Hp * self.inputs, 1)
        else:
            self.disturbance = disturbance
        self.solver_input = mpc.gen_solver_input(self.initial_state, self.u_prev, self.ref, self.disturbance)

        # Save constraints
        # State bounds
        if lower_bounds_states is None:
            self.lower_bounds_states = ca.DM.zeros(self.states * self.Hp, 1) - ca.inf
        else:
            self.lower_bounds_states = lower_bounds_states

        if upper_bounds_states is None:
            self.upper_bounds_states = ca.DM.zeros(self.states * self.Hp, 1) + ca.inf
        else:
            self.upper_bounds_states = upper_bounds_states

        # Slew rate bounds
        if lower_bounds_slew_rate is None:
            self.lower_bounds_slew_rate = ca.DM.zeros(self.inputs * self.Hp, 1) - ca.inf
        else:
            self.lower_bounds_slew_rate = lower_bounds_slew_rate

        if upper_bounds_slew_rate is None:
            self.upper_bounds_slew_rate = ca.DM.zeros(self.inputs * self.Hp, 1) + ca.inf
        else:
            self.upper_bounds_slew_rate = upper_bounds_slew_rate

        # Input bounds
        if lower_bounds_input is None:
            self.lower_bounds_input = ca.DM.zeros(self.inputs * self.Hp, 1) - ca.inf
        else:
            self.lower_bounds_input = lower_bounds_input

        if upper_bounds_input is None:
            self.upper_bounds_input = ca.DM.zeros(self.inputs * self.Hp, 1) + ca.inf
        else:
            self.upper_bounds_input = upper_bounds_input

        lbx = self.lower_bounds_slew_rate
        ubx = self.upper_bounds_slew_rate
        lbg = ca.vertcat(ca.vec(self.lower_bounds_states), ca.vec(self.lower_bounds_input))
        ubg = ca.vertcat(ca.vec(self.upper_bounds_states), ca.vec(self.upper_bounds_input))

        self.result = self.solver(p=self.solver_input, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
        self.dU = self.result['x']
        self.prev_steps = self.dU
        # self.prev_steps.append(self.result)
        self.predicted_states = mpc.gen_predicted_states(self.psi, self.initial_state, self.upsilon, self.u_prev, self.theta,
                                                         self.dU, self.theta_d, self.disturbance)

        self.log_length = log_length
        self.log_initial_state = ca.vec(initial_state)
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

    def set_dynamics_matrix(self, dynamics_matrix):
        self.lift(dynamics_matrix=dynamics_matrix)

    def get_dynamics_matrix(self):
        return self.dynamics_matrix

    def set_input_matrix(self, input_matrix):
        self.lift(input_matrix=input_matrix)

    def get_input_matrix(self):
        return self.input_matrix

    def set_input_matrix_d(self, input_matrix_d):
        self.lift(input_matrix_d=input_matrix_d)

    def get_input_matrix_d(self):
        return self.input_matrix_d

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

    def log(self, initial_state, expected_x, dU, ref, disturbance):

        self.log_initial_state = ca.horzcat(self.log_initial_state, ca.vec(initial_state))
        self.log_ref = ca.horzcat(self.log_ref, ca.vec(ref))
        self.log_disturbance = ca.horzcat(self.log_disturbance, ca.vec(disturbance))
        self.log_expected_x = ca.horzcat(self.log_expected_x, ca.vec(expected_x))
        self.log_dU = ca.horzcat(self.log_dU, ca.vec(dU))

    def get_expected_x_log(self):

        return self.log_expected_x

    def get_initial_state_log(self):

        return self.log_initial_state

    def get_ref_log(self):

        return self.log_ref

    def get_dU_log(self):

        return self.log_dU

    def lift(self, dynamics_matrix=None, input_matrix=None, Hu=None, Hp=None, Q=None, R=None, input_matrix_d=None):

        if dynamics_matrix != None and dynamics_matrix.shape == self.dynamics_matrix.shape:
            self.dynamics_matrix = dynamics_matrix
        if input_matrix != None and input_matrix.shape == self.input_matrix.shape:
            self.input_matrix = input_matrix
        if input_matrix_d != None and input_matrix_d.shape == self.input_matrix_d.shape:
            self.input_matrix_d = input_matrix_d
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

        self.psi = mpc.gen_psi(self.dynamics_matrix, self.Hp)
        self.upsilon = mpc.gen_upsilon(self.dynamics_matrix, self.input_matrix, self.Hp)
        self.theta = mpc.gen_theta(self.upsilon, self.input_matrix, self.Hu)
        self.theta_d = mpc.gen_theta(self.upsilon, self.input_matrix_d, Hu)
        self.solver = mpc.gen_mpc_solver(self.dynamics_matrix, self.input_matrix, self.Hu, self.Hp, self.Qb, self.Rb, self.input_matrix_d)

    def print_result(self):
        print(self.result)

    def print_solver(self):
        print(self.solver)

    def step(self, initial_state, u_prev, ref, disturbance=None):
        self.initial_state = initial_state
        self.u_prev = u_prev
        self.ref = ref
        if disturbance is None:
            self.disturbance = ca.DM.zeros(self.Hp * self.inputs, 1)
        else:
            self.disturbance = disturbance
        self.solver_input = mpc.gen_solver_input(self.initial_state, self.u_prev, self.ref, self.disturbance)
        self.result = self.solver(p=self.solver_input)

        self.dU = self.result['x']
        self.prev_steps = self.dU
        # self.prev_steps.append(self.result)
        self.predicted_states = mpc.gen_predicted_states(self.psi, self.initial_state, self.upsilon, self.u_prev, self.theta,
                                                         self.dU, self.theta_d, self.disturbance)
        self.log(initial_state, self.get_next_expected_state(), self.get_next_control_input_change(), self.get_next_ref(),
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
            plt.plot(t[0:2], ca.vertcat(self.initial_state[s], self.predicted_states[s]), 'r-')
            plt.plot(t[0], self.initial_state[s], '.', color=col, label="State {}={:.3f}".format(s, float(self.initial_state[s])))

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
                plt.plot(t[1:self.k + 1], self.get_initial_state_log()[s, 0:].T, color=col)
                plt.plot(t[2:self.k + 2], self.get_expected_x_log()[s, 0:].T, color=col, ls='-.')

                plt.plot(t[self.k + 1:], self.predicted_states[s::self.states], color=col, alpha=0.6)
                plt.plot(t[self.k + 1:], self.ref[s::self.states], '--', color=col, alpha=0.6)
                plt.plot(t[2:self.k + 2], self.get_ref_log()[s, :].T, '--', color=col)
                plt.plot(t[self.k:self.k + 2], ca.vertcat(self.initial_state[s], self.predicted_states[s]), 'r-')
                plt.plot(t[self.k], self.initial_state[s], '.', color=col, label="State {}={:.3f}".format(s, float(self.initial_state[s])))

        plt.ylabel('States')
        ax1.legend()
        ax2 = plt.subplot(212, sharex=ax1)

        for i in range(0, self.inputs):
            col = plt.cm.tab10(i)
            w = 'post'
            if i not in ignore_inputs:
                if opts['drawU'] == 'both' or opts['drawU'] == 'U':
                    # plt.plot(t[1:self.k + 1], self.get_initial_state_log()[s, 0:].T, color=col)

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
    #     x_prediction = mpc.gen_predicted_states(self.psi, self.initial_state, self.upsilon,
    #                                             self.u_prev, self.theta, )
    #     self.solver_input = mpc.gen_solver_input(initial_state, u_prev, ref)
    #     self.result = self.solver(p=self.solver_input)
