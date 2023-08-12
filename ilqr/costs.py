import numpy as np


class AbstractCost:
    def eval(self, x, u):
        raise NotImplementedError

    def dx(self, x, u):
        raise NotImplementedError

    def du(self, x, u):
        raise NotImplementedError

    def dxu(self, x, u):
        raise NotImplementedError

    def dux(self, x, u):
        raise NotImplementedError

    def dxx(self, x, u):
        raise NotImplementedError

    def duu(self, x, u):
        raise NotImplementedError


class QuadraticCost(AbstractCost):
    def __init__(self, state_position_cost, state_velocity_cost, action_cost, target_state):
        self.x_target = target_state
        self.Q = np.diag(np.array(state_position_cost) + np.array(state_velocity_cost))
        self.R = np.diag(np.array(action_cost))

    def eval(self, x, u):
        x_diff = x - self.x_target
        x_loss = np.dot(x_diff.T, np.dot(self.Q, x_diff))
        u_loss = np.dot(u.T, np.dot(self.R, u))
        return x_loss + u_loss

    def dx(self, x, u):
        x_diff = x - self.x_target
        return 2 * np.dot(self.Q, x_diff)

    def du(self, x, u):
        return 2 * np.dot(self.R, u)

    def dxu(self, x, u):
        return np.zeros_like(x)

    def dux(self, x, u):
        return np.zeros_like(x)

    def dxx(self, x, u):
        return 2 * self.Q

    def duu(self, x, u):
        return 2 * self.R
