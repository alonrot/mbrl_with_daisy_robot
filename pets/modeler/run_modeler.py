# sys.path.append(os.getcwd())
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import visdom
from omegaconf import OmegaConf
from scipyplot.plot import rplot

import modeler.functions as f
from mbrl import utils
from mbrl.models import EnsembleModel
from modeler import modeler_utils

vis = visdom.Visdom()


class Data:
    """
    Keeps raw data
    specifically, a sequence of x points, of function mean and of variance at each x
    as well as points samples around mean within variance at each x
    """

    def __init__(self, cfg, function, variance):
        self.function = function
        # x axis values
        self.x = np.arange(cfg.x_start, cfg.x_end, (cfg.x_end - cfg.x_start) / cfg.dataset_size, dtype=np.float32)
        # mean and variance
        self.mean = np.array([self.function(i) for i in self.x], dtype=np.float32)
        self.var = np.array([variance(i) for i in self.x], dtype=np.float32)
        # sample y values from mean and variance
        stddev = np.sqrt(self.var)
        self.y = self.mean + np.random.normal(0, stddev, self.mean.shape).astype(dtype=np.float32)


class Solver:

    def __init__(self, cfg):
        type = cfg.type.upper()
        tconf = cfg.types[type]

        def model_factory():
            tconf.model.params.device = cfg.device
            return utils.instantiate(tconf.model)

        def loss_factory():
            return utils.instantiate(tconf.loss)

        self.model = EnsembleModel(model_factory, loss_factory, cfg.ensemble_size, cfg.device)

    def train(self, data, cfg):
        dataset = Dataset(data.x, data.y)
        self.model.train_model(training_dataset=dataset, testing_dataset=None, cfg=cfg)
        visualize(cfg, self.model, data, dataset, cfg.vis_x_points)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.dataset = [(x[i], y[i]) for i in range(x.shape[0])]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def dataset_points(dataset):
    # returns all data points in the dataset, sorted by x value
    xs = []
    ys = []
    for x, y in dataset:
        xs.append(x)
        ys.append(y)

    npx = np.array(xs)
    npy = np.array(ys)
    xind = npx.argsort()
    sorted_x = npx[xind[::-1]]
    sorted_y = npy[xind[::-1]]
    return sorted_x, sorted_y


def visualize(cfg, model, data, dataset, steps):
    xs = np.linspace(cfg.x_start, cfg.x_end, num=steps, endpoint=True)
    if model.is_deterministic():
        visualize_deterministic(model, data, dataset, xs)
    else:
        visualize_probabilistic(model, data, dataset, xs)


def visualize_probabilistic(model, data, dataset, x_values):
    device = model.device
    assert device is not None
    name = data.function.output_dim_name
    x = torch.from_numpy(np.array([i for i in x_values], dtype=np.float32))
    if x.dim() == 1:
        x = x.unsqueeze(0).t()
    x = x.to(device=device)

    y = torch.Tensor([data.function(i) for i in x_values]).float()

    for d in range(y.size(1)):
        means = []
        variances = []
        for eid in range(model.ensemble_size):
            mean_and_var = model.models[eid](x).detach().cpu().numpy()
            pmean = mean_and_var[0]
            pvar = mean_and_var[1]
            means.append(pmean)
            variances.append(pvar)

            fig = plt.figure()
            data_x, data_y = dataset_points(dataset)
            plt.plot(data_x, data_y[:, d], 'k*')
            fig = rplot(fig=fig, x=data.x, y=data.mean[:, d], uncertainty=data.var[:, d],
                        color=['green'])
            fig = rplot(fig=fig, x=x.cpu().numpy(), y=pmean[:, d], uncertainty=pvar[:, d],
                        legend=[f"{name(d)} EID:{eid}"],
                        color=['red'])

            vis.matplot(fig, win=f"{eid}_r-plot %{name(d)}")
            plt.close()

        mean, var = utils.moment_matching(means, variances)
        fig = plt.figure()
        fig = rplot(fig=fig, x=x.cpu().numpy(), y=mean[:, d], uncertainty=var[:, d],
                    legend=[f"{name(d)} MM"],
                    color=['blue'])
        vis.matplot(fig, win=f"mm-r-plot %{name(d)}")
        plt.close()


def visualize_deterministic(model, data, dataset, x_values):
    name = data.function.output_dim_name
    x = torch.from_numpy(np.array([i for i in x_values], dtype=np.float32))
    if x.dim() == 1:
        x = x.unsqueeze(0).t()
    gt_y = torch.from_numpy(np.array([data.function(i) for i in x_values])).float()
    for eid in range(model.ensemble_size):
        data_x, data_y = dataset_points(dataset)
        pred_y = model.models[eid](x).detach().cpu()
        for d in range(gt_y.squeeze().dim()):
            ys = torch.stack((gt_y[:, d], pred_y[:, d])).numpy()

            fig = plt.figure()
            plt.plot(data_x, data_y[:, d], 'k*')
            rplot(fig=fig, x=x.numpy(), y=ys[0], color=['green'])
            rplot(fig=fig, x=x.numpy(), y=ys[1], color=['red'])

            vis.matplot(fig, win=f"{eid}_func_{name(d)}_rplot")
            plt.close()


def main():
    modeler_utils.config_log()
    conffile = OmegaConf.load('modeler/config.yaml')
    cfg = OmegaConf.merge(conffile, OmegaConf.from_cli())
    modeler_utils.random_seed(cfg.seed)
    cfg.type = cfg.type.upper()
    configs = dict(
        D=dict(
            # 1D deterministic function
            data=Data(
                cfg,
                function=f.Sin1D(),
                variance=f.Constant(0)
            ),
        ),
        D2=dict(
            # 2D deterministic function
            data=Data(
                cfg,
                # function=lambda x: np.array([math.sin(x), math.cos(x)]),
                function=f.SinCos2D(),
                variance=f.Constant(0, 0)
            ),
        ),
        P=dict(
            # 1D noisy function
            data=Data(
                cfg,
                function=f.Sin1D(),
                variance=f.Constant(0.1)
            ),
        ),
        P2=dict(
            # 2D noisy function with different noise on each dimension
            data=Data(
                cfg,
                function=f.SinCos2D(),
                variance=f.Constant(0.1, 0.2)
            ),
        ),
    )

    experiment = configs.get(cfg.type, None)
    if experiment is None:
        raise Exception(f"Unsupported type {cfg.type}, should be one of {list(configs.keys())}")

    print("Config :")
    print(cfg.pretty())

    solver = Solver(cfg)
    data = experiment['data']
    solver.train(data, cfg)


if __name__ == '__main__':
    main()
