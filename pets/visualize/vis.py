import logging
import math
import re
from collections import defaultdict
from enum import Enum

import dictdiffer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from scipyplot.plot import rplot, rplot_data
from torch.utils.data import DataLoader

# noinspection PyUnresolvedReferences
import mbrl.environments
from mbrl import utils
from mbrl.dataset import SASDataset
from mbrl.dynamics_model import EnvBasedDynamicsModel
from mbrl.policies import RandomPolicy
from mbrl.trajectories.utils import compute_trajectories, gather_actions

log = logging.getLogger(__name__)


def filter_logs(all_logs, diffs, filter_regex=None):
    if filter_regex is None:
        return all_logs

    ret = defaultdict(list)
    for log_dir, diff in diffs.items():
        desc = ",".join([f"{key}={value}" for key, value in diffs[log_dir]])
        if len(re.findall(filter_regex, desc)) > 0:
            log.info(f"Dir {log_dir} : passed filter : {desc}")
            ret[log_dir] = all_logs[log_dir]

    return ret


def create_config_diffs(all_configs, exclude_keys=[]):
    # take first config from each experiment as a representative
    configs = {}
    for log_dir in all_configs.keys():
        spec = all_configs[log_dir][0].spec
        configs[log_dir] = spec.to_container() if spec else {}
    if len(configs) <= 1:
        # single config, nothing to diff.
        return defaultdict(list)

    first_dir = list(all_configs.keys())[0]
    raw_diffs = {}
    for log_dir in all_configs.keys():
        raw_diffs[log_dir] = list(dictdiffer.diff(configs[first_dir], configs[log_dir], ignore=exclude_keys))

    def expand_add(value, root):
        ret = []
        if isinstance(value, dict):
            for dict_key, dict_key in value.items():
                ret.extend(expand_add(dict_key, f"{root}.{dict_key}"))
        else:
            ret.append((root, value))

        return ret

    def expand_add_keys(add_remove_diff):
        assert add_remove_diff[0] in ['add', 'remove'], f"Unexpected diff : {add_remove_diff}"
        ret = []
        for add in add_remove_diff[2]:
            ret.extend(expand_add(add[1], add[0]))
        return ret

    diffs = defaultdict(dict)
    for log_dir in configs.keys():
        for diff in raw_diffs[log_dir]:
            if diff[0] == 'add':
                for k, v in expand_add_keys(diff):
                    diffs[log_dir][k] = v
            elif diff[0] == 'remove':
                for k, v in expand_add_keys(diff):
                    diffs[first_dir][k] = f"{v}"
            elif diff[0] == 'change':
                key = diff[1]
                change = diff[2]
                diffs[first_dir][key] = change[0]
                diffs[log_dir][key] = change[1]
            else:
                log.error(f"Unexpected change type {diff}")

    for key, value in diffs[first_dir].items():
        for log_dir, diff in diffs.items():
            if key not in diff.keys():
                diff[key] = value

    # compact keys by removing common prefixes
    different_keys = list(next(iter(diffs.values())).keys())
    if len(different_keys) > 0:
        first_key = different_keys[0]
        parts = first_key.split('.')
        for prefix in parts:
            all_starts_with = True
            for log_dir in diffs.keys():
                for key in diffs[log_dir].keys():
                    if not key.startswith(f"{prefix}."):
                        all_starts_with = False
                        break
            if all_starts_with:
                for log_dir in diffs.keys():
                    diffres = {}
                    for key, value in diffs[log_dir].items():
                        new_key = key[len(prefix) + 1:]
                        if new_key is not '':
                            diffres[new_key] = value
                    diffs[log_dir] = diffres

    # convert dictionary to sorted list
    res = defaultdict(list)
    for log_dir, diff in diffs.items():
        changes = [(key, value) for key, value in diffs[log_dir].items()]
        changes.sort(key=lambda x: x[0])
        res[log_dir] = changes

    return res


def plot_rewards_over_trials(vis_obj, vis_env, all_logs, diffs):
    data = []
    traces = []
    colors = plt.get_cmap('tab10').colors

    for i, (log_dir, logs) in enumerate(all_logs.items()):
        if len(diffs[log_dir]) > 0:
            string = ",".join([f"{e[0]}={e[1]}" for e in diffs[log_dir]])
        else:
            string = log_dir
        cs_str = 'rgb' + str(colors[i])
        if i == 0: env_name = logs[0]['env_name']

        ys = np.stack([np.asarray(log['rewards']) for log in logs])
        data.append(ys)
        err_traces, xs, ys = generate_errorbar_traces(np.asarray(data[i]), color=cs_str, name=f"{string}")
        for t in err_traces:
            traces.append(t)

    layout = dict(title=f"Learning Curve Reward vs Number of Trials (Env: {env_name})",
                  xaxis={'title': 'Trial Num'},
                  yaxis={'title': 'Cum. Reward'},
                  font=dict(family='Times New Roman', size=30, color='#7f7f7f'),
                  height=1000,
                  width=1500,
                  legend={'x': .83, 'y': .05, 'bgcolor': 'rgba(50, 50, 50, .03)'})

    vis_obj._send({'data': traces, 'layout': layout, 'win': f"err_plot_debug", 'eid': vis_env})  # 'layout': layout,


def generate_errorbar_traces(ys, xs=None, percentiles='66+95', color=None, name=None):
    if xs is None:
        xs = [list(range(len(y))) for y in ys]

    minX = min([len(x) for x in xs])

    xs = [x[:minX] for x in xs]
    ys = [y[:minX] for y in ys]

    assert all([(len(y) == len(ys[0])) for y in ys]), \
        'Y should be the same size for all traces'

    assert all([(x == xs[0]) for x in xs]), \
        'X should be the same for all traces'

    y = np.array(ys)

    def median_percentile(data, des_percentiles='66+95'):
        median = np.nanmedian(data, axis=0)
        out = np.array(list(map(int, des_percentiles.split("+"))))
        for i in range(out.size):
            assert 0 <= out[i] <= 100, 'Percentile must be >0 <100; instead is %f' % out[i]
        list_percentiles = np.empty((2 * out.size,), dtype=out.dtype)
        list_percentiles[0::2] = out  # Compute the percentile
        list_percentiles[1::2] = 100 - out  # Compute also the mirror percentile
        percentiles = np.nanpercentile(data, list_percentiles, axis=0)
        return [median, percentiles]

    out = median_percentile(y, des_percentiles=percentiles)
    ymed = out[0]
    # yavg = np.median(y, 0)

    err_traces = [
        dict(x=xs[0], y=ymed.tolist(), mode='lines', name=name, type='line', legendgroup=f"group-{name}",
             line=dict(color=color, width=4))]

    intensity = .3
    '''
    interval = scipy.stats.norm.interval(percentile/100, loc=y, scale=np.sqrt(variance))
    interval = np.nan_to_num(interval)  # Fix stupid case of norm.interval(0) returning nan
    '''

    for i, p_str in enumerate(percentiles.split("+")):
        p = int(p_str)
        high = out[1][2 * i, :]
        low = out[1][2 * i + 1, :]

        err_traces.append(dict(
            x=xs[0] + xs[0][::-1], type='line',
            y=(high).tolist() + (low).tolist()[::-1],
            fill='toself',
            fillcolor=(color[:-1] + str(f", {intensity})")).replace('rgb', 'rgba')
            if color is not None else None,
            line=dict(color='transparent'),
            legendgroup=f"group-{name}",
            showlegend=False,
            name=name + str(f"_std{p}") if name is not None else None,
        ), )
        intensity -= .1

    return err_traces, xs, ys


def plot_dynamics_model_tests(vis_env, filtered_logs, configs, diffs):
    """
    :param checkpoint: File to load from /checkpoint/usr/runs/ with dynamics model and training dataset
    :param policy: policy to plan with for trajectory estimates
    :param vis_env: page to return plots on visdom (default 'dynam')
    :return:
    """
    import gym

    # TODO
    #   1. support multiple log files
    #   2. compile outputs from ensemble models nicely
    #   3. convert to visdom
    #   4. incorporate variance predictions from probablistic models
    #   5. figure out how to load the policy that was used at a rollout

    for log_dir, logs in filtered_logs.items():
        # get name
        if len(diffs[log_dir]) > 0:
            title = ",".join([f"{e[0]}={e[1]}" for e in diffs[log_dir]])
        else:
            title = log_dir
        cfg = configs[log_dir][0]

        env_name = logs[0]['env_name']
        ys = np.stack([np.asarray(log['rewards']) for log in logs])

        # load checkpoint items
        dynam_model = logs[0]['dynamics_model']
        training_dataset = logs[0]['training_dataset']
        testing_dataset = logs[0]['testing_dataset']
        groundtruth_ep = logs[0]['episode']
        groundtruth_traj = torch.stack([s.s0 for s in groundtruth_ep], dim=0)

        # generate arrays
        states = torch.stack([s for s in training_dataset.states0], dim=0)
        groundtruth_pred = torch.stack([s for s in training_dataset.states1], dim=0)
        actions = torch.stack([a for a in training_dataset.actions], dim=0)

        # prediction one step dynamics
        predictions = dynam_model.predict(states, actions)
        state0 = groundtruth_traj[0, :]  # states[0, :]
        device = state0.device

        # convert for plotting
        predictions = predictions.cpu().numpy()
        groundtruth_pred = groundtruth_pred.cpu().numpy()

        plot_one_step_ahead_predictions(predictions[:, :, 0], groundtruth_pred, vis_env, title)

        env = gym.make(cfg.env.name)
        policy = utils.instantiate(cfg.policy, cfg)
        policy.setup(dynam_model, env.action_space, utils.get_static_method(cfg.env.reward_func))

        def gen_plans(true_trajectory, policy, dynam_model):
            time = true_trajectory.shape[0]
            ds = true_trajectory.shape[1]
            da = policy.cfg.env.action_size
            plans = torch.empty((time, policy.planning_horizon + 1, ds))
            for i, state in enumerate(true_trajectory):
                action_seqs = policy.plan_action_sequence(state)
                plans[i, :, :] = compute_trajectories(dynam_model, state,
                                                      action_seqs.actions.reshape(1, policy.planning_horizon, da)
                                                      ).squeeze()

            return plans

        def to_np(tensor):
            return tensor.cpu().detach().numpy()

        plans = gen_plans(groundtruth_traj, policy, dynam_model)
        plot_planning(to_np(plans), to_np(groundtruth_traj), vis_env)

        # 3. plot predicted trajectories of planned actions through model
        N = 10
        da = actions.shape[1]
        ds = state0.shape[0]

        # adjust policy to longer prediction horizon
        cfg.policy.params.planning_horizon = groundtruth_traj.shape[0]
        policy = utils.instantiate(cfg.policy, cfg)
        policy.setup(dynam_model, env.action_space, utils.get_static_method(cfg.env.reward_func))

        actions_long = gather_actions(policy.optimizer, N, state0, da).to(device)
        predictions_traj = compute_trajectories(dynam_model, state0,
                                                actions_long.reshape(N, policy.planning_horizon, da)
                                                ).squeeze()
        plot_trajectory_pred(to_np(predictions_traj[:, :-1, :]), to_np(groundtruth_traj), vis_env)


def plot_one_step_ahead_predictions(vis_obj, predictions, groundtruth, vis_env, title='One Step Model Predictions',
                                    cmap=cm.tab10):
    """
    This function plot the predictions of the model against the groundtruth state.
    For visualization purposes, it also sorts the data w.r.t. groundtruth.
    (by doing so, we get as a bonus also the cdf of the groundtruth)
    :param predictions: np.array of dimension [N.Data x N.States]
    :param groundtruth: np.array of dimension [N.Data x N.States]
    :return:
    """
    assert groundtruth.shape == predictions.shape, 'Wrong dimensions'

    for i, d in enumerate(range(groundtruth.shape[1])):
        opts = dict(title=title,
                    font=dict(family='Times New Roman', size=18, color='#7f7f7f'),
                    showlegend=True,
                    xlabel='Sorted Ground Truth',
                    ylabel=f"State Dimension - {i}",
                    # legend=['Predictions', 'Ground Truth'],
                    linecolor=np.array([[int(c * 255) for c in cmap(1)], [int(c * 255) for c in cmap(0)]]), linewidth=3,
                    win=f"dim-{i}")

        idx = np.argsort(groundtruth[:, d])
        vis_obj.line(Y=np.hstack([predictions[idx, d:d + 1], groundtruth[idx, d:d + 1]]),
                     name=['Predictions', 'Ground Truth'], env=vis_env + "_pred",
                     opts=opts)


def plot_planning_mpl(vis_obj, plans, groundtruth, vis_env, title='', cmap=cm.viridis):
    """
    Plots trajectories replanned at each timestep given a ground truth trajectory. Use this for lower memory impact
    :param plans: np.array of dimension [Time x Horizon x N.States]
    :param groundtruth: np.array of dimension [Time x N.States]
    :param cmap: matplotlib cmap
    :return:
    """
    n_curves = plans.shape[0]
    h = plans.shape[1]
    T = groundtruth.shape[0]
    dim_state = groundtruth.shape[1]

    colors = cmap(np.linspace(0, 0.85, n_curves))  # Removing final 15% of the viridis colormap

    fig_list = []
    for i, d in enumerate(range(dim_state)):
        fig = plt.figure()
        for it, t in enumerate(range(n_curves)):
            y = np.arange(it, it + h)
            plt.plot(y, plans[it, :, d], color=colors[it], linewidth=1)
        plt.plot(groundtruth[:, d], color='black', linewidth=2, label='groundtruth')
        plt.ylabel('Variable %d' % i)
        plt.xlabel('Time')
        plt.show()
        fig_list.append(fig)

    for i, fig in enumerate(fig_list):
        win = f"dim_{i}_policy_plans"
        vis_obj.matplot(fig, win=win, env=vis_env)


def plot_planning(vis_obj, plans, groundtruth, vis_env, filter_num=3):
    """
    Plots trajectories replanned at each timestep given a ground truth trajectory
    :param plans: np.array of dimension [Time x Horizon x N.States]
    :param groundtruth: np.array of dimension [Time x N.States]
    :param cmap: matplotlib cmap
    :return:
    """

    n_curves = plans.shape[0]
    h = plans.shape[1]
    T = groundtruth.shape[0]
    dim_state = groundtruth.shape[1]

    for i, d in enumerate(range(dim_state)):
        traces = []
        for it, t in enumerate(range(n_curves)):
            if (it % filter_num != 0): continue
            y = np.arange(it, it + h).tolist()

            estimated_trace_pt = dict(
                x=y,
                y=plans[it, :, d].tolist(),
                type='scatter',
                mode='markers',
                marker=dict(color=np.arange(h).tolist(), colorscale='Viridis', size=6),
                showlegend=(it == 1),
                legendgroup=f"trajs-{d}",
                name='Planned Trajectory',
            )
            traces.append(estimated_trace_pt)

            estimated_trace_line = dict(
                x=y,
                y=plans[it, :, d].tolist(),
                type='line',
                mode='lines',
                line=dict(color='rgba(100,100,100,.3)', width=1),
                showlegend=False,
                legendgroup=f"trajs-{d}",
            )
            traces.append(estimated_trace_line)

        truth_trace = dict(
            x=np.arange(T).tolist(),
            y=groundtruth[:, d].tolist(),
            type='line',
            showlegend=True,
            line=dict(color='black', width=4),
            name='True Episode',
        )
        traces.append(truth_trace)

        layout = dict(title=f"Estimated Trajectories from Each Episode Step (Dim {d})",
                      xaxis={'title': 'Time (Step)'},
                      yaxis={'title': 'State Value'},
                      font=dict(family='Times New Roman', size=18, color='#7f7f7f'),
                      legend={'x': .83, 'y': .05, 'bgcolor': 'rgba(50, 50, 50, .03)'},
                      # width=1200,
                      # height=700,
                      )

        vis_obj._send({'data': traces, 'layout': layout, 'win': f"traj-dim-{d}", 'eid': vis_env + "_est"})


def plot_trajectory_pred(vis_obj, predictions, groundtruth, vis_env, title='State Trajectories Predicted via Optimizer',
                         cmap=cm.tab10):
    """
    Plots predicted trajectories from initial state for given number of trajectories
    :param predictions:  np.array of dimension [Time x N.States x N.Predictions]
    :param groundtruth: np.array of dimension [Time x N.States]
    :param title:
    :param cmap:
    :return:
    """
    dim_state = groundtruth.shape[1]

    linecolors = [[int(c * 255) for c in cmap(1)] for i in range(predictions.shape[2])]
    linecolors.append([int(c * 255) for c in cmap(0)])
    linecolors = np.array(linecolors)

    for i, d in enumerate(range(dim_state)):
        opts = dict(title=title,
                    font=dict(family='Times New Roman', size=18, color='#7f7f7f'),
                    # showlegend=False,
                    xlabel='Time (points)',
                    ylabel=f"State Dimension - {i}",
                    linecolor=linecolors,
                    linewidth=[2],
                    # legend=['Predictions', 'Ground Truth'],
                    win=f"dim-{i}")

        data = np.hstack([predictions[n, :, d:d + 1] for n in range(predictions.shape[2])])
        data = np.hstack([data, groundtruth[:, d:d + 1]])

        vis_obj.line(Y=data, X=np.arange(groundtruth.shape[0]),
                     name=['Predictions', 'Ground Truth'], env=vis_env + "_long",
                     opts=opts)


def plot_train_and_test_loss(vis_obj, vis_env, all_logs, diffs):
    train_losses = []
    test_losses = []
    legend = []
    for log_dir, logs in all_logs.items():
        env_name = logs[0]['env_name']
        for log in logs:
            log_train_losses = [training_log.train_loss for training_log in log['training_logs']]
            train_losses.append(np.array([log_train_losses]))

            log_test_losses = [training_log.test_loss for training_log in log['training_logs']]
            test_losses.append(np.array([log_test_losses]))

        if len(diffs[log_dir]) > 0:
            desc = ",".join([f"{e[0]}={e[1]}" for e in diffs[log_dir]])
        else:
            desc = log_dir
        legend.append(desc)

    percentiles = [66]
    win = f"group_{env_name}_train_loss"
    fig = rplot_data(data=train_losses,
                     distribution=f'median+{"+".join(str(x) for x in percentiles)}',
                     legend=legend,
                     xlabel="train loss")
    vis_obj.matplot(fig, win=f"{win}_{env_name}_rplot", env=vis_env)
    plt.clf()
    win = f"group_{env_name}_test_loss"
    fig = rplot_data(data=test_losses,
                     distribution=f'median+{"+".join(str(x) for x in percentiles)}',
                     legend=legend,
                     xlabel="test loss")
    vis_obj.matplot(fig, win=f"{win}_{env_name}_rplot", env=vis_env)
    plt.clf()


def visualize_group(vis_obj, vis_env, all_logs, all_configs, filter_regex=None):
    if len(all_logs) == 0:
        return

    diffs = create_config_diffs(all_configs, ['full_log_dir',
                                              'random_seed',
                                              'device',
                                              'instance_id',
                                              'log_dir',
                                              'log_dir_suffix'])

    filtered_logs = filter_logs(all_logs, diffs, filter_regex)

    plot_rewards_over_trials(vis_obj, vis_env, filtered_logs, diffs)


def visualize_dynamics(vis_obj, vis_env, all_logs, all_configs, filter_regex=None):
    if len(all_logs) == 0:
        print("No log files passed, exiting plotter")
        return

    diffs = create_config_diffs(all_configs, ['full_log_dir',
                                              'random_seed',
                                              'device',
                                              'instance_id',
                                              'log_dir',
                                              'log_dir_suffix'])

    filtered_logs = filter_logs(all_logs, diffs, filter_regex)

    plot_dynamics_model_tests(vis_obj, vis_env, filtered_logs, all_configs, diffs)


def visualize_reward(vis_obj, vis_env, log):
    env_name = log['env_name']
    y = torch.from_numpy(np.asarray(log['rewards']))
    x = torch.arange(len(y))
    title = f'{env_name}: reward per trial'
    win = f"{env_name}_win_reward"
    opts = dict(
        title=title,
        legend=['Rewards'],
        xlabel='Reward',
        ylabel='Trial number',
    )
    vis_obj.line(Y=y, X=x, win=win, env=vis_env, opts=opts)


def visualize_loss(vis_obj, vis_env, log):
    env_name = log['env_name']
    trial_num = log['trial_num']
    train_loss = torch.FloatTensor(log['train_log'].mean_epoch_train_loss)
    test_loss = torch.FloatTensor(log['train_log'].mean_epoch_test_loss)
    x = torch.arange(train_loss.size(0))
    legend = []
    ys = [train_loss]
    legend.append('Train loss')
    if test_loss is not None and len(test_loss) > 0:
        ys.append(test_loss)
        legend.append('Test loss')
    y = torch.stack(ys)
    title = '{}:trial {} loss'.format(env_name, trial_num)
    win = f"{env_name}_win_loss"
    opts = dict(
        title=title,
        legend=legend,
        xlabel='Epoch',
        ylabel='Training loss',
    )
    vis_obj.line(Y=y.t(), X=x, win=win, env=vis_env, opts=opts)


def test_and_visualize(vis_obj, vis_env, cfg, trial_num, log, delta=True, sort=True):
    """
    Generate new samples using the real environment, and compare predictions of the learned dynamics model to the
    ground truth generated by the real environment
    :param vis_env:
    :param cfg:
    :param trial_num:
    :param log:
    :return:
    """
    num_episodes = 5
    task_horizon = 200
    gt_model = EnvBasedDynamicsModel(cfg.env.name)
    dynamics_model = log['dynamics_model']
    random_policy = RandomPolicy(cfg.device, gt_model.env.action_space, cfg.mpc.planning_horizon)

    dataset = SASDataset()
    for i in range(num_episodes):
        ep = utils.sample_episode(gt_model.env, random_policy, task_horizon)
        dataset.add_episode(ep.episode, cfg.device)

    loader = DataLoader(dataset, batch_size=num_episodes * task_horizon)
    _idx, data = next(enumerate(loader))
    actions = data['action']
    states = data['state']
    preds = dynamics_model.predict(states, actions)
    gt_target = data['new_state']

    visualize_test_results_impl(vis_obj, vis_env,
                                cfg.env.name,
                                dynamics_model.is_probabilistic(),
                                trial_num,
                                states,
                                gt_target,
                                preds,
                                delta, sort)


def visualize_test_results_impl(vis_obj, env_name, is_probablistic, trial_num, states, gt_target, preds, delta=True,
                                sort=True):
    d = gt_target.size(1)
    gt_target = gt_target.cpu().float()
    states = states.cpu().float()
    preds = preds.cpu().float()

    # ignore variance in probabilistic models for now and just plot mean
    if is_probablistic:
        mean = preds[0]
        var = preds[1]
    else:
        mean = preds
        var = torch.zeros(preds.shape)

    for dim in range(d):
        var_col = var[:, dim]
        pred_col = mean[:, dim]
        gt_target_col = gt_target[:, dim]

        if delta:
            state_col = states[:, dim]
            pred_col = pred_col - state_col
            gt_target_col = gt_target_col - state_col

        if sort:
            gt_target_col, order = gt_target_col.sort(0)
            pred_col = pred_col.index_select(0, order)
            var_col = var_col.index_select(0, order)

        y = torch.stack((pred_col, gt_target_col))
        x = torch.arange(pred_col.size(0))
        title = '{}:trial {}:dim {}'.format(env_name, trial_num, dim)
        win = f"{env_name}_win_{dim}"

        opts = dict(
            title=title,
            legend=['Predicted', 'GT'],
        )
        # vis.line(Y=y.t(), X=x, win=win, env=vis_env, opts=opts)

        fig = plt.figure()
        plt.plot(x.numpy(), gt_target_col.numpy(), 'k*')
        fig = rplot(fig=fig, y=pred_col.numpy(), uncertainty=var_col.numpy(), color=['green'],
                    legend=[f"Mean and var on dim {dim}"])
        fig = rplot(fig=fig, x=x.numpy(), y=gt_target_col.numpy(), color=['red'])

        vis_obj.matplot(fig, win=f"{win}_rplot")
        plt.clf()


class DimeDesc(Enum):
    SCALAR = 1
    ANGLE_DEGREES = 2
    ANGLE_RADIANS = 3


def visualize_model_dims_sweep(vis_obj, env_name, dynamics_model, dim_desc=[]):
    """
    :param env_name: environment name
    :param dynamics_model: dynamics model
    :param dim_desc: description of dimentions
    :return:
    """
    gt_model = EnvBasedDynamicsModel(env_name)
    state = gt_model.env.reset()
    action = gt_model.env.action_space.sample()
    for i in range(state.shape[0]):
        x = None
        if dim_desc[i] == DimeDesc.ANGLE_DEGREES:
            x = np.arange(-180 * 1.5, 180 * 1.5, 1)
        elif dim_desc[i] == DimeDesc.ANGLE_RADIANS:
            x = np.arange(-math.pi * 3, math.pi * 3, 0.1)

        if x is not None:
            states = np.tile(state, (x.shape[0], 1))
            actions = np.tile(action, (x.shape[0], 1))
            states[:, i] = x
            gt_new_states = gt_model.predict(states, actions)[:, i]
            new_states = dynamics_model.predict(states, actions)[:, i]
            y = np.stack((new_states, gt_new_states), 1)
            x = np.stack((x, x), 1)
            vis_obj.line(Y=y, X=x, win=f"win_{i}", opts=dict(title=f'{env_name}_dim_{i}', legend=['Predicted', 'GT']))
