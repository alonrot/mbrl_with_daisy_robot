import numpy as np
import torch
import visdom
from omegaconf import OmegaConf

import modeler as mu
import modeler.functions as f
from mbrl import utils

vis = visdom.Visdom()


def plot(cfg, func):
    dx = np.abs(cfg.vis.x1 - cfg.vis.x2)
    dy = np.abs(cfg.vis.y1 - cfg.vis.y2)
    xs = np.linspace(cfg.vis.x1, cfg.vis.x2, num=dx / cfg.vis.step, endpoint=True)
    ys = np.linspace(cfg.vis.y1, cfg.vis.y2, num=dy / cfg.vis.step, endpoint=True)
    zs = np.zeros((len(xs), len(ys)))
    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            z = func(np.array([x, y]))
            zs[xi, yi] = z
    vis.surf(zs, win=func.__repr__())


class Optimizer:
    def optimize(self, *args, **kwargs):
        raise NotImplementedError("Must be implemented in subclass.")


class CEM(Optimizer):

    def __init__(self, cfg):
        self.popsize = cfg.popsize
        self.num_elites = cfg.num_elites
        self.max_iters = cfg.max_iters
        self.mean_alpha = cfg.mean_alpha
        self.var_alpha = cfg.var_alpha
        self.epsilon = cfg.epsilon
        self.debug = cfg.debug

    def optimize(self, function, lower_bound, upper_bound, initial_mean, initial_variance, minimize=True):
        assert torch.is_tensor(initial_mean)
        assert torch.is_tensor(initial_variance)
        assert torch.is_tensor(lower_bound)
        assert torch.is_tensor(upper_bound)

        mean = initial_mean
        var = initial_variance
        input_size = function.input_size()
        n = 0
        means = []
        variances = []
        samples = torch.zeros(self.popsize, input_size)

        while n < self.max_iters and var.max().item() > self.epsilon:
            lb_dist = mean - lower_bound
            ub_dist = upper_bound - mean
            mv = torch.min(torch.pow(lb_dist / 2, 2), torch.pow(ub_dist / 2, 2))
            constrained_var = torch.min(mv, var)
            samples = utils.truncated_normal_(samples) * torch.sqrt(constrained_var) + mean

            costs = function(samples.t()).t()
            best_costs, costs_index = torch.topk(costs, k=self.num_elites, dim=0, largest=not minimize)
            elites = samples.index_select(dim=0, index=costs_index)

            new_mean = torch.mean(elites, dim=0)
            new_var = torch.var(elites, unbiased=False, dim=0)

            mean = self.mean_alpha * mean + (1 - self.mean_alpha) * new_mean
            var = self.var_alpha * var + (1 - self.var_alpha) * new_var
            means.append(mean)
            variances.append(var)

            if self.debug:
                print(f"Iter {n}, value={function(new_mean)}")
            n += 1
        diag = dict(var=var, means=means, variances=variances)
        return dict(result=mean, diag=diag)


def main():
    mu.config_log()
    conffile = OmegaConf.load('modeler/optimizers/cem.yaml')
    cfg = OmegaConf.merge(conffile, OmegaConf.from_cli())

    a = 2
    b = 10

    mu.random_seed(cfg.seed)
    func = f.TorchBanana(a, b)
    initial_mean = torch.from_numpy(np.tile((func.lower_bound() + func.upper_bound()) / 2, [1])).float()
    initial_var = torch.from_numpy(np.tile(np.square(func.lower_bound() - func.upper_bound()) / 16, [1])).float()
    optimizer = CEM(cfg.cem)

    lower_bound = torch.from_numpy(func.lower_bound()).float()
    upper_bound = torch.from_numpy(func.upper_bound()).float()

    solution = optimizer.optimize(func, lower_bound, upper_bound, initial_mean, initial_var)

    plot(cfg, func)

    result = solution['result']
    minp = np.array([a, a * a])
    print("Analytical global minimum", minp, func(minp))
    print("Computed minimum", result, func(result))


if __name__ == '__main__':
    main()
