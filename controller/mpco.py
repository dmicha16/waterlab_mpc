from controller import mpc
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


class MpcObj:

    def __init__(self, dynamics_matrix, input_matrix, control_horizon, prediction_horizon, state_cost,
                 input_change_cost,
                 ref=None, initial_control_signal=None, input_matrix_d=None,
                 lower_bounds_states=None, upper_bounds_states=None,
                 lower_bounds_slew_rate=None, upper_bounds_slew_rate=None,  # TODO slew
                 lower_bounds_input=None, upper_bounds_input=None):

        """
        :param dynamics_matrix:(states x states) Model dynamics matrix of type casadi.DM
        :param input_matrix:(states x inputs) Input dynamics matrix of type casadi.DM
        :param control_horizon: (int) Control horizon of type Integer
        :param prediction_horizon: (int) Prediction horizon of type Integer
        :param state_cost:(states x states) State cost matrix of type casadi.DM
        :param input_change_cost:(inputs x inputs) Input change cost matrix of type casadi.DM
        :param initial_control_signal:(inputs x 1) Initial control input for Log
        :param ref: (states * control_horizon x 1) input reference trajectory
        :param input_matrix_d: ( disturbances x prediction_horizon ) Input dynamic matrix for disturbances
        :param log_length: (int) Nr. of steps to log, 0 = all TODO implement functionality
        :param lower_bounds_states: (states*prediction_horizon x 1) Lower bounds vector for states as type casadi.DM
        :param upper_bounds_states: (states*prediction_horizon x 1) Upper bounds vector for states as type casadi.DM
        :param lower_bounds_slew_rate: (inputs*control_horizon x 1) Lower bounds vector for slew rates as type casadi.DM
        :param upper_bounds_slew_rate: (inputs*control_horizon x 1) Upper bounds vector for slew rates as type casadi.DM
        :param lower_bounds_input: (inputs*control_horizon x 1) Lower bounds vector for inputs as type casadi.DM
        :param upper_bounds_input: (inputs*control_horizon x 1) Lower bounds vector for slew rates as type casadi.DM
        """
        # Getting number of states and inputs from matrix dimensions
        self.states = dynamics_matrix.size1()
        self.inputs = input_matrix.size2()
        self.disturbances = input_matrix_d.size2()

        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon

        # handle None inputs
        if input_matrix_d is None:
            self.input_matrix_d = input_matrix
        else:
            self.input_matrix_d = input_matrix_d

        # State bounds
        if lower_bounds_states is None:
            self.lower_bounds_states = ca.DM.zeros(self.states * self.prediction_horizon, 1) - ca.inf
        else:
            self.lower_bounds_states = lower_bounds_states

        if upper_bounds_states is None:
            self.upper_bounds_states = ca.DM.zeros(self.states * self.prediction_horizon, 1) + ca.inf
        else:
            self.upper_bounds_states = upper_bounds_states

            # Slew rate bounds
        if lower_bounds_slew_rate is None:
            self.lower_bounds_slew_rate = ca.DM.zeros(self.inputs * self.prediction_horizon, 1) - ca.inf
        else:
            self.lower_bounds_slew_rate = lower_bounds_slew_rate

        if upper_bounds_slew_rate is None:
            self.upper_bounds_slew_rate = ca.DM.zeros(self.inputs * self.prediction_horizon, 1) + ca.inf
        else:
            self.upper_bounds_slew_rate = upper_bounds_slew_rate

            # Input bounds
        if lower_bounds_input is None:
            self.lower_bounds_input = ca.DM.zeros(self.inputs * self.prediction_horizon, 1) - ca.inf
        else:
            self.lower_bounds_input = lower_bounds_input

        if upper_bounds_input is None:
            self.upper_bounds_input = ca.DM.zeros(self.inputs * self.prediction_horizon, 1) + ca.inf
        else:
            self.upper_bounds_input = upper_bounds_input

        if ref is None:
            self.ref = ref = ca.DM.zeros(self.prediction_horizon * self.states, 1)
        else:
            self.ref = ref

        if initial_control_signal is None:
            self.u_prev = ca.DM.zeros(self.inputs)
        else:
            self.u_prev = initial_control_signal

        # Save dynamics and solver variables
        self.dynamics_matrix = dynamics_matrix
        self.input_matrix = input_matrix
        self.state_cost = state_cost
        self.state_cost_block_matrix = mpc.blockdiag(state_cost, prediction_horizon)
        self.input_change_cost = input_change_cost
        self.input_change_cost_block_matrix = mpc.blockdiag(input_change_cost, control_horizon)
        self.psi = mpc.gen_psi(dynamics_matrix, prediction_horizon)
        self.upsilon = mpc.gen_upsilon(dynamics_matrix, input_matrix, prediction_horizon)
        self.theta = mpc.gen_theta(self.upsilon, input_matrix, control_horizon)

        self.upsilon_d = mpc.gen_upsilon(dynamics_matrix, input_matrix_d, prediction_horizon)
        self.theta_d = mpc.gen_theta(self.upsilon_d, self.input_matrix_d, prediction_horizon)
        self.solver = self.lift(dynamics_matrix, input_matrix, control_horizon, prediction_horizon, state_cost,
                                input_change_cost, input_matrix_d)

        self.result = None
        # Setup step variables
        self.k = 0
        self.initial_state = ca.DM.zeros(self.states)
        self.ref = ref

        self.disturbance = ca.DM.zeros(self.prediction_horizon * self.disturbances, 1)
        self.prev_disturbance = ca.DM.zeros(self.inputs, 1)
        self.solver_input = mpc.gen_solver_input(self.initial_state, self.u_prev, self.ref, self.prev_disturbance,
                                                 self.disturbance)
        self.predicted_states = ca.DM.zeros(self.prediction_horizon * self.states, 1)
        self.dU = ca.DM.zeros(self.control_horizon * self.inputs, 1)

        # Constraints
        self.lbx = self.lower_bounds_slew_rate
        self.ubx = self.upper_bounds_slew_rate
        self.lbg = ca.vertcat(ca.vec(self.lower_bounds_states), ca.vec(self.lower_bounds_input))
        self.ubg = ca.vertcat(ca.vec(self.upper_bounds_states), ca.vec(self.upper_bounds_input))

        # Logs
        self.log_initial_state = ca.vec(self.initial_state)
        self.log_ref = ca.vec(self.get_next_ref())
        self.log_disturbance = ca.vec(self.get_next_disturbance_change())
        self.log_expected_x = ca.vec(self.get_next_expected_state())
        self.log_dU = ca.vec(self.u_prev)
        self.log_control_cost = ca.vec(self.get_input_change_cost() @ self.get_input_change_cost() @ self.u_prev)
        self.log_state_cost = ca.vec(self.get_state_cost() @ self.get_state_cost() @ self.initial_state)

    def get_next_ref(self):
        return self.ref[:self.states]

    def get_next_disturbance_change(self):
        return self.disturbance[:self.disturbances]

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

    def set_control_horizon(self, control_horizon):
        self.lift(control_horizon=control_horizon)

    def get_control_horizon(self):
        return self.control_horizon

    def set_prediction_horizon(self, prediction_horizon):
        self.lift(prediction_horizon=prediction_horizon)

    def get_prediction_horizon(self):
        return self.prediction_horizon

    def set_state_cost(self, state_cost):
        self.lift(state_cost=state_cost)

    def get_state_cost(self):
        return self.state_cost

    def set_input_change_cost(self, input_change_cost):
        self.lift(input_change_cost=input_change_cost)

    def get_input_change_cost(self):
        return self.input_change_cost

    def get_du(self):
        return self.dU

    def log(self, initial_state, expected_x, dU, ref, disturbance):
        self.log_initial_state = ca.horzcat(self.log_initial_state, ca.vec(initial_state))
        self.log_ref = ca.horzcat(self.log_ref, ca.vec(ref))
        self.log_disturbance = ca.horzcat(self.log_disturbance, ca.vec(disturbance))
        self.log_expected_x = ca.horzcat(self.log_expected_x, ca.vec(expected_x))
        self.log_dU = ca.horzcat(self.log_dU, ca.vec(dU))
        self.log_control_cost = ca.horzcat(self.log_control_cost,
                                           ca.vec(self.get_input_change_cost() @ self.get_input_change_cost() @ dU))
        self.log_state_cost = ca.horzcat(self.log_state_cost,
                                         ca.vec(self.get_state_cost() @ self.get_state_cost() @ initial_state))

    def get_expected_x_log(self):
        return self.log_expected_x[:, 1:]

    def get_initial_state_log(self):
        return self.log_initial_state[:, 1:]

    def get_state_cost_log(self):
        return self.log_state_cost[:, 1:]

    def get_control_cost(self):
        return self.log_control_cost[:, 1:]

    def get_ref_log(self):
        return self.log_ref[:, 1:]

    def get_dU_log(self):
        return self.log_dU[:, 0:]

    def lift(self, dynamics_matrix=None, input_matrix=None, control_horizon=None, prediction_horizon=None,
             state_cost=None, input_change_cost=None, input_matrix_d=None):
        # TODO: update
        if dynamics_matrix is not None and dynamics_matrix.shape == self.dynamics_matrix.shape:
            self.dynamics_matrix = dynamics_matrix
        if input_matrix is not None and input_matrix.shape == self.input_matrix.shape:
            self.input_matrix = input_matrix
        if input_matrix_d is not None and input_matrix_d.shape == self.input_matrix_d.shape:
            self.input_matrix_d = input_matrix_d
        if prediction_horizon is not None:
            self.prediction_horizon = prediction_horizon
        if control_horizon is not None:
            self.control_horizon = control_horizon
        if state_cost is not None and state_cost.shape == self.state_cost.shape:
            self.state_cost = state_cost
            self.state_cost_block_matrix = mpc.blockdiag(state_cost, prediction_horizon)
        if input_change_cost is not None and input_change_cost.shape == self.state_cost.shape:
            self.input_change_cost = input_change_cost
            self.input_change_cost_block_matrix = mpc.blockdiag(input_change_cost, control_horizon)

        self.psi = mpc.gen_psi(self.dynamics_matrix, self.prediction_horizon)
        self.upsilon = mpc.gen_upsilon(self.dynamics_matrix, self.input_matrix, self.prediction_horizon)
        self.upsilon_d = mpc.gen_upsilon(self.dynamics_matrix, self.input_matrix_d, self.prediction_horizon)
        self.theta = mpc.gen_theta(self.upsilon, self.input_matrix, self.control_horizon)
        self.theta_d = mpc.gen_theta(self.upsilon_d, self.input_matrix_d, control_horizon)
        self.solver = mpc.gen_mpc_solver(self.dynamics_matrix, self.input_matrix, self.control_horizon,
                                         self.prediction_horizon,
                                         self.state_cost_block_matrix, self.input_change_cost_block_matrix,
                                         self.input_matrix_d)
        return self.solver

    def print_result(self):
        print(self.result)

    def print_solver(self):
        print(self.solver)

    def step(self, initial_state, u_prev, ref=None, prev_disturbance=None, disturbance=None):
        if disturbance is None:
            self.disturbance = ca.DM.zeros(self.prediction_horizon * self.disturbances, 1)
        else:
            self.disturbance = disturbance

        if prev_disturbance is None:
            self.prev_disturbance = ca.DM.zeros(self.disturbances, 1)
        else:
            self.prev_disturbance = prev_disturbance

        if ref is not None:
            self.ref = ref

        self.initial_state = initial_state
        self.u_prev = u_prev

        self.solver_input = mpc.gen_solver_input(self.initial_state, self.u_prev, self.ref, self.prev_disturbance,
                                                 self.disturbance)
        self.result = self.solver(p=self.solver_input, lbg=self.lbg, ubg=self.ubg, lbx=self.lbx, ubx=self.ubx)

        self.dU = self.result['x']
        self.predicted_states = mpc.gen_predicted_states(self.psi, self.initial_state, self.upsilon, self.u_prev,
                                                         self.theta,
                                                         self.dU, self.upsilon_d, self.prev_disturbance, self.theta_d,
                                                         self.disturbance)
        self.log(initial_state, self.get_next_expected_state(), self.get_next_control_input_change(),
                 self.get_next_ref(),
                 self.get_next_disturbance_change(), )
        self.k = self.k + 1

    def plot_step(self, options={}, plot_pause=0.3):
        # should plot
        #   input_change_costeference
        #   Predicted States
        #   Inputs
        #   (Costs: dU og State cost)
        if self.k == 0:
            print('Step before plotting')
            return

        plt.clf()

        opts = {'drawU': 'U'}
        opts.update(options)

        t = np.arange(0, self.get_prediction_horizon() + 1)

        ax1 = plt.subplot(211)
        plt.title("Optimized step at time k = {}".format(self.k))
        for s in range(0, self.states):
            col = plt.cm.tab10(s)

            plt.plot(t[1:], self.predicted_states[s::self.states], color=col)
            plt.plot(t[1:], self.ref[s::self.states], '--', color=col)
            plt.plot(t[0:2], ca.vertcat(self.initial_state[s], self.predicted_states[s]), 'r-')
            plt.plot(t[0], self.initial_state[s], '.', color=col,
                     label="State {}={:.3f}".format(s, float(self.initial_state[s])))

        plt.ylabel('States')
        ax1.legend()
        ax2 = plt.subplot(212, sharex=ax1)

        t = np.append(np.arange(-1, 0), t)
        for i in range(0, self.inputs):
            col = plt.cm.tab10(i)
            w = 'post'

            if opts['drawU'] == 'both' or opts['drawU'] == 'U':
                U = np.cumsum(ca.vertcat(self.u_prev[i], self.dU[i::self.inputs]))

                plt.step(t[0:self.control_horizon + 1], U, color=col, where=w)
                #       plt.step(t[1:3], U[1:3], color=col, where=w)
                plt.plot(t[1], U[1], '.', color=col, label="u{}={:.3f}".format(i, float(U[1])))

                plt.step(t[1:3], U[0:2], 'r-', where="pre")

            if opts['drawU'] == 'both' or opts['drawU'] == 'dU':
                idU = self.dU[i::self.inputs]
                plt.plot(t[1:self.control_horizon + 1], idU, '.', color=col)
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
        #   input_change_costeference
        #   Predicted States
        #   Inputs
        #   (Costs: dU og State cost)
        if self.k == 0:
            print('Step before plotting')
            return

        plt.clf()

        opts = {'drawU': 'U'}
        opts.update(options)

        t = np.arange(-self.k, self.get_prediction_horizon() + 1)

        ax1 = plt.subplot(211)
        plt.title("Progress at time k = {}".format(self.k))
        plt.xlim([-self.k, self.prediction_horizon])

        for s in range(0, self.states):
            if s not in ignore_states:
                col = plt.cm.tab10(s)
                plt.plot(t[1:self.k + 1], self.get_initial_state_log()[s, 0:].T, color=col)
                plt.plot(t[2:self.k + 2], self.get_expected_x_log()[s, 0:].T, color=col, ls='-.')

                plt.plot(t[self.k + 1:], self.predicted_states[s::self.states], color=col, alpha=0.6)
                plt.plot(t[self.k + 1:], self.ref[s::self.states], '--', color=col, alpha=0.6)
                plt.plot(t[2:self.k + 2], self.get_ref_log()[s, :].T, '--', color=col)
                plt.plot(t[self.k:self.k + 2], ca.vertcat(self.initial_state[s], self.predicted_states[s]), 'r-')
                plt.plot(t[self.k], self.initial_state[s], '.', color=col,
                         label="State {}={:.3f}".format(s, float(self.initial_state[s])))

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

                    plt.step(t[0:self.k + self.control_horizon], U, color=col, where=w)
                    #       plt.step(t[1:3], U[1:3], color=col, where=w)
                    plt.plot(t[self.k], U[self.k], '.', color=col, label="U{}={:.3f}".format(i, float(U[self.k])))

                    plt.step(t[self.k:self.k + 2], U[self.k - 1:self.k + 1], 'r-', where="pre")

                if opts['drawU'] == 'both' or opts['drawU'] == 'dU':
                    idU = ca.vertcat(self.get_dU_log()[i, 0:-1].T, self.dU[i::self.inputs])

                    plt.plot(t[0:self.k + self.control_horizon], idU, '.', color=col)
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
