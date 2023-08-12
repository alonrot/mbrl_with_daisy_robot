import logging
import numpy as np
import torch

from mbrl.policies import Policy
from mbrl.optimizers import ActionSequence
from mbrl.dataset import SAS, SASDataset

log = logging.getLogger(__name__)


class LocallyLinearFeedbackActionSequence(ActionSequence):
    def __init__(self, Ks, trajectory: SASDataset):
        super(LocallyLinearFeedbackActionSequence, self).__init__()
        self.Ks = Ks
        self.xs = np.array([sas.s0.numpy() for sas in trajectory])
        self.us = np.array([sas.a.numpy() for sas in trajectory])

    def get_action(self, x, t):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        action = self.us[t] + self.Ks[t].dot(x - self.xs[t])
        return torch.Tensor(action)


class iLQRPolicy(Policy):
    def __init__(self,
                 state_dimension,
                 action_dimension,
                 dynamics_model,
                 cost,
                 time_horizon,
                 sigma=0,  # Curious iLQR parameter. 0 implies the usual iLQR.
                 dt=1/240.0,
                 ):
        self.dX = state_dimension
        self.dU = action_dimension
        self.last_trajectory = None
        self._policy = None
        self._action_sequence = None

        self._dynamics = dynamics_model
        self._time_horizon = time_horizon
        self._cost = cost
        self._reg_min = 1e-6
        self._reg_factor = 10
        self._reg_max = 1000

        # Regularization terms: Levenberg-Marquardt parameter.
        # See II F. Regularization Schedule.
        self._mu = 1.0
        self._mu_min = 1e-6
        self._mu_max = 1e4
        self._delta_0 = 2.0
        self._delta = self._delta_0
        self._sigma = sigma
        self._dt = dt

    def _forward_pass(self, x0, u_seq):
        x_traj = np.zeros((self._time_horizon + 1, len(x0)))
        x_traj[0, :] = x0
        for t, u in enumerate(u_seq):
            x_traj[t + 1, :] = self._dynamics.predict(x_traj[t, :], u)

        return x_traj

    @staticmethod
    def _smooth_inv(m):
        try:
            w, v = np.linalg.eigh(m)
            w_inv = w / (w ** 2 + 1e-6)
            return v.dot(np.diag(w_inv)).dot(v.transpose())
        except:
            return np.linalg.inv(m)

    @staticmethod
    def _regularize(m):
        w, v = np.linalg.eigh(m)
        if w[0] < 3e-2:
            m += (abs(w[0]) + 3e-2) * np.eye(len(w))
        return np.linalg.inv(m)

    def _backward_pass(self, x_traj, u_seq):
        '''
        :param x_traj: states of shape [time_horizon + 1, dim state]
        :param u_seq: actions of shape [time_horizon, dim action]
        :return:
        '''

        k = [None] * self._time_horizon
        K = [None] * self._time_horizon

        Vxx = self._cost.dxx(x_traj[-1, :], u_seq[-1])
        Vx = self._cost.dx(x_traj[-1, :], u_seq[-1])

        for t in reversed(range(self._time_horizon)):
            xt, ut = x_traj[t, :], u_seq[t, :]
            q, r = self._cost.dx(xt, ut), self._cost.du(xt, ut)
            Q, R = self._cost.dxx(xt, ut), self._cost.duu(xt, ut)
            P = self._cost.dux(xt, ut)

            A = self._dynamics.dx_fd(xt, ut)
            B = self._dynamics.du_fd(xt, ut)

            sigma = self._sigma

            C = 0.03 * np.eye(self.dX)
            C[:self.dU, :self.dU] = self._dt * np.eye(self.dU)

            _, uncertainty = self._dynamics.predict_with_uncertainty(xt, ut)
            covar = np.diag(uncertainty)

            covar = np.clip(covar, 0.0, 0.75)
            term_covar = C.dot(covar).dot(C.T).round(decimals=3)

            H = R + B.T.dot(Vxx).dot(B) + sigma * B.T.dot(Vxx.T).dot(term_covar).dot(Vxx).dot(B)

            g = r + B.T.dot(Vx) + sigma * B.T.dot(Vxx.T).dot(term_covar).dot(Vx)

            G = P.T + B.T.dot(Vxx).dot(A) + sigma * B.T.dot(Vxx.T).dot(term_covar.T).dot(Vxx).dot(A)

            iH = self._smooth_inv(H)

            k[t] = -iH.dot(g)
            K[t] = -iH.dot(G)

            s = q + A.T.dot(Vx) + G.T.dot(k[t]) + K[t].T.dot(g) + K[t].T.dot(H).dot(k[t]) + sigma * A.T.dot(Vxx.T).dot(
                term_covar).dot(Vx)

            Vx = s[:]

            S = Q + A.T.dot(Vxx).dot(A) + K[t].T.dot(H).dot(K[t]) + G.T.dot(K[t]) + K[t].T.dot(G) + sigma * A.T.dot(
                Vxx.T).dot(term_covar).dot(Vxx).dot(A)
            Vxx = 0.5 * (S + S.T)[:]

        return np.array(k), np.array(K)

    def _eval_trajectory_cost(self, x_traj, u_seq):
        J = 0.0
        for t in range(self._time_horizon):
            xt, ut = x_traj[t, :], u_seq[t]
            J += self._cost.eval(xt, ut)

        J += self._cost.eval(x_traj[-1, :], np.zeros_like(ut))
        return J

    def _run_feedback_policy(self, x_des, u_seq, k, K, alpha):
        x_new = np.zeros_like(x_des)
        u_new = np.zeros_like(u_seq)

        x_new[0, :] = x_des[0, :]
        for t in range(self._time_horizon):
            u_new[t] = u_seq[t] + alpha * k[t] + K[t].dot(x_new[t] - x_des[t])
            x_new[t + 1] = self._dynamics.predict(x_new[t], u_new[t])

        return x_new, u_new

    def iteration_callback(self, iteration_count, xs, us, J_opt, accepted, converged):
        info = 'converged' if converged else ('accepted' if accepted else 'failed')
        final_state = xs[-1]
        log.info(f'iterations {iteration_count} {info} {J_opt} {final_state[:7]}')

    def _run(self, x0, u_init_seq, max_iter, debug=True, tol=1e-4):
        self._mu = 1.0
        alphas = 1.1 ** (-np.arange(10) ** 2)

        u_seq = u_init_seq.copy()
        x_traj = self._forward_pass(x0, u_init_seq)
        k_accepted = np.zeros_like(u_seq)
        K_accepted = np.zeros((u_seq.shape[0], u_seq.shape[1], len(x0)))

        J_opt = self._eval_trajectory_cost(x_traj, u_init_seq)

        converged = False
        for i in range(max_iter):
            accept = False
            k, K = self._backward_pass(x_traj, u_seq)

            for alpha in alphas:
                x_traj_new, u_seq_new = self._run_feedback_policy(x_traj, u_seq, k, K, alpha)
                J_new = self._eval_trajectory_cost(x_traj_new, u_seq_new)
                if J_new < J_opt:
                    log.info(f'selected alpha = {alpha}')
                    if np.abs((J_opt - J_new) / J_opt) < tol:
                        converged = True
                        break
                    else:
                        # decrease regularization
                        self._delta = min(1.0, self._delta) / self._delta_0
                        self._mu *= self._delta
                        if self._mu <= self._mu_min:
                            self._mu = 0.0
                        accept = True

                    J_opt = J_new
                    x_traj = x_traj_new.copy()
                    u_seq = u_seq_new.copy()
                    K_accepted = K.copy()
                    k_accepted = k.copy()
                    break
                else:
                    accept = False

            if not accept:
                self._delta = max(1.0, self._delta) * self._delta_0
                self._mu = max(self._mu_min, self._mu * self._delta)
                self.iteration_callback(i, x_traj, u_seq, J_opt, accept, converged)
                if self._mu_max and self._mu >= self._mu_max:
                    log.info('regularization term too large, quitting')
                break

            if converged:
                log.info('converged')
                break

            if debug:
                self.iteration_callback(i, x_traj, u_seq, J_opt, accept, converged)
        self._policy = lambda x, t: u_seq[t] + K_accepted[t].dot(x - x_traj[t])

        # Create ActionSequence usable for sampling.
        trajectory = SASDataset()
        for t in range(self._time_horizon):
            s0 = torch.Tensor(x_traj[t])
            a = torch.Tensor(u_seq[t])
            s1 = torch.Tensor(x_traj[t+1])
            sas = SAS(s0, a, s1)
            trajectory.add(sas)
        self._action_sequence = LocallyLinearFeedbackActionSequence(Ks=K_accepted, trajectory=trajectory)

    def optimize(self, trajectory: SASDataset, max_iter):
        x0 = trajectory[0].s0
        u_init_seq = np.array([sas.a.numpy() for sas in trajectory])
        self._run(x0, u_init_seq, max_iter)

    def plan_action_sequence(self, state) -> ActionSequence:
        assert self._action_sequence is not None, 'Must call optimize before planning policy'
        return self._action_sequence
