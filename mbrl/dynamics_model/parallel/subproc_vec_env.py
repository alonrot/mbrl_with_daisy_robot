import logging
from multiprocessing import Process, Pipe

import numpy as np

log = logging.getLogger(__name__)


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


# make synchronous interface for get call
def worker(remote, parent_remote, env):
    parent_remote.close()
    env = env.x() if hasattr(env, 'x') else env()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                ob, reward, done, info = env.step(data)
                if done:
                    ob = env.reset()
                remote.send((ob, reward, done, info))
            elif cmd == 'get':
                remote.send(getattr(env, data))
            elif cmd == 'close':
                remote.close()
                break  # this terminates the process.
            else:
                if type(data) == tuple:
                    _ = getattr(env, cmd)(*data)
                else:
                    data = data or dict()
                    args = data.get('args', tuple())
                    kwargs = data.get('kwargs', dict())
                    _ = getattr(env, cmd)(*args, **kwargs)
                remote.send(_)

        except EOFError as e:  # process has ended from inside
            break  # this terminates the process
        except KeyboardInterrupt as e:  # process has ended from inside
            break  # this terminates the process
        except BaseException as e:
            log.error(f"{type(e)} : {e}")
            break


class SubprocVecEnv:
    reset_on_done = True

    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(self.num_envs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.first = self.remotes[0]
        self.first.send(('get', 'action_space'))
        self.action_space = self.first.recv()
        self.first.send(('get', 'observation_space'))
        self.observation_space = self.first.recv()
        self.first.send(('get', 'spec'))
        self.spec = self.first.recv()

    def call_sync(self, fn_name, *args, **kwargs):
        _ = fn_name, dict(args=args, kwargs=kwargs)
        for remote in self.remotes:
            remote.send(_)
        try:
            return np.stack([remote.recv() for remote in self.remotes])
        except EOFError as e:
            raise RuntimeError('Unknown Error has occurred with the environment.') from e

    def get(self, key):
        raise NotImplementedError('need to decide for self.first or all.')

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        obs, rews, dones, infos = zip(*[remote.recv() for remote in self.remotes])
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def render(self, *args, **kwargs):
        return self.call_sync('render', *args, **kwargs)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def set_state_batch(self, qpos_batch, qvel_batch):
        assert (len(qpos_batch) == len(qvel_batch))
        assert (len(qpos_batch) == len(self.remotes))

        for i, remote in enumerate(self.remotes):
            remote.send(('set_state', (qpos_batch[i], qvel_batch[i])))

        return [remote.recv() for remote in self.remotes]

    def compute_next_state_batch(self, states, actions):
        assert (len(states) == len(actions))
        assert (len(states) == len(self.remotes))

        for i, remote in enumerate(self.remotes):
            remote.send(('compute_next_state', (states[i], actions[i])))

        return [remote.recv() for remote in self.remotes]

    def compute_next_states_batch(self, state_chunks, action_chunks):
        assert (type(state_chunks) == list)
        assert (type(action_chunks) == list)
        assert (len(state_chunks) == len(action_chunks))
        num_batches = len(state_chunks)

        for i in range(num_batches):
            self.remotes[i].send(('compute_next_states', (state_chunks[i], action_chunks[i])))

        results = []
        for i in range(num_batches):
            results.append(self.remotes[i].recv())
        return results

    def reset(self):
        return self.call_sync('reset')

    def reset_task(self):
        self.call_sync('reset_task')

    def close(self):
        """looks bad: mix sync and async handling."""
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True
