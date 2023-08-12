import ctypes
from time import time

import gym
import numpy as np
# noinspection PyUnresolvedReferences
import torch
from omegaconf import OmegaConf

from mbrl import utils, optimizers
from mbrl.dynamics_model import NNBasedDynamicsModel
from mbrl.models import EnsembleModel

"""
Execute with nvprof:

time PYTHONPATH=. /usr/local/cuda/bin/nvprof --cpu-thread-tracing on --output-profile /tmp/profile.nvvp -f --profile-from-start off --track-memory-allocations on --trace gpu,api python benchmark/compute_trajectories.py

You can choose different benchmark classes at the bottom section

"""

_cudart = ctypes.CDLL('libcudart.so')


def prof_start():
    # As shown at http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
    # the return value will unconditionally be 0. This check is just in case it changes in
    # the future.
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)


def prof_stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)


planning_horizon = 30
num_trajectories = 2500
num_particles = 20


class Benchmark:
    def run_benchmark(self, n):
        raise NotImplemented()


k = 5
w = 500


class BenchBMM(Benchmark):
    def __init__(self):
        tensors = [torch.ones(w, w).cuda() for _ in range(k)]
        self.tensors2 = torch.stack(tensors)
        self.output = torch.zeros(w, w).cuda()

    def run_benchmark(self, n):
        for i in range(n):
            torch.bmm(self.tensors2, self.tensors2, out=self.output)


class BenchSerial(Benchmark):
    def __init__(self):
        self.tensors = [torch.ones(w, w).cuda() for _ in range(k)]
        self.output = torch.zeros(w, w).cuda()

    def run_benchmark(self, n):
        for i in range(n):
            for inp in self.tensors:
                torch.matmul(inp, inp, out=self.output)


class BenchStreams(Benchmark):
    def __init__(self):
        self.tensors = [torch.ones(w, w).cuda() for _ in range(k)]
        self.output = torch.zeros(w, w).cuda()
        self.streams = [torch.cuda.Stream() for _ in range(k)]

    def run_benchmark(self, n):
        for i in range(n):
            for idx in range(k):
                with torch.cuda.stream(self.streams[idx]):
                    torch.matmul(self.tensors[idx], self.tensors[idx], out=self.output)

            for idx in range(k):
                torch.cuda.current_stream().wait_stream(self.streams[idx])


class BenchForward(Benchmark):
    def __init__(self):
        cfg = OmegaConf.load("benchmark/compute_trajectories.yaml")
        type_str = cfg.dynamics_model.type.upper()
        params = cfg.dynamics_model[type_str]

        def model_factory():
            return utils.instantiate(params, 24, 16, cfg.device)

        def loss_factory():
            return utils.instantiate(params.loss)

        self.input1 = torch.zeros(num_trajectories, 24).to(device=cfg.device)
        self.model = EnsembleModel(model_factory, loss_factory, cfg.dynamics_model.ensemble_size, cfg.device)

    def run_benchmark(self, n):
        for i in range(n):
            self.model.forward(self.input1)


import queue
import threading

start = time()


class BenchForwardMT(Benchmark):
    class ForwardWorker(threading.Thread):
        def __init__(self, input_queue, output_queue):
            super(BenchForwardMT.ForwardWorker, self).__init__()
            self.setDaemon(True)
            self.input_queue = input_queue
            self.output_queue = output_queue

        def run(self):
            global start
            while True:
                if not self.input_queue.empty():
                    item = self.input_queue.get()
                    if item is None:
                        break
                    metadata, func = item
                    t = time()
                    output = func()
                    # print(f"{metadata} {time() - start} took {time() - t}, putting result")
                    self.output_queue.put((metadata, output))

    def __init__(self):
        cfg = OmegaConf.load("benchmark/compute_trajectories.yaml")
        type_str = cfg.dynamics_model.type.upper()
        params = cfg.dynamics_model[type_str]

        def model_factory():
            return utils.instantiate(params, 24, 16, cfg.device)

        def loss_factory():
            return utils.instantiate(params.loss)

        self.input1 = torch.zeros(num_trajectories, 24).to(device=cfg.device)
        self.model = EnsembleModel(model_factory, loss_factory, cfg.dynamics_model.ensemble_size, cfg.device)

        buffer_size = cfg.dynamics_model.ensemble_size * 2
        self.input_queue = queue.Queue(buffer_size)
        self.output_queue = queue.Queue(buffer_size)
        self.workers = []
        for _ in range(cfg.dynamics_model.ensemble_size):
            worker = BenchForwardMT.ForwardWorker(self.input_queue, self.output_queue)
            worker.start()
            self.workers.append(worker)

    def run_benchmark(self, n):
        for i in range(n):

            def forward(models, inputs):
                assert len(models) == len(models)

                results = []

                t = time()
                global start
                start = t
                # print(f"{t - start} Starting")
                # submit one forward for each model to workers
                for eid in range(len(models)):
                    results.append(None)

                    def func():
                        return models[eid](inputs[eid])

                    self.input_queue.put((eid, func))
                t = time()
                # print(f"{t - start} Jobs submitted")
                # fetch one result from output queue
                for _ in range(len(models)):
                    eid, result = self.output_queue.get()
                    # print(f"{eid} {time() - start} : got result")
                    results[eid] = result
                # print(f"Results received {t - start}")
                return results

            self.model.forward(self.input1, forward)


class BenchForwardMT2(Benchmark):
    class ForwardWorker(threading.Thread):
        def __init__(self, index, models, inputs, outputs, input_cond, output_event):
            super(BenchForwardMT2.ForwardWorker, self).__init__()
            self.setDaemon(True)
            self.setName(f"Forward worker {index}")
            self.index = index
            self.models = models
            self.inputs = inputs
            self.outputs = outputs
            self.input_cond = input_cond
            self.output_event = output_event
            self.running = True

        def run(self):
            while self.running:
                with self.input_cond:
                    # print(f"Thread {self.getName()} waiting")
                    self.input_cond.wait()
                if not self.running:
                    break

                # print(f"Thread {self.getName()} woke up, processing input {self.index}")
                self.outputs[self.index] = self.models[self.index].forward(self.inputs[self.index])

                # print(f"Thread {self.getName()} setting event")
                self.output_event.set()

        def stop(self):
            self.running = False

    def __init__(self):
        cfg = OmegaConf.load("benchmark/compute_trajectories.yaml")
        type_str = cfg.dynamics_model.type.upper()
        params = cfg.dynamics_model[type_str]

        def model_factory():
            return utils.instantiate(params, 24, 16, cfg.device)

        def loss_factory():
            return utils.instantiate(params.loss)

        self.input_cond = threading.Condition()
        self.output_events = [threading.Event() for _ in range(cfg.dynamics_model.ensemble_size)]
        self.input1 = torch.zeros(num_trajectories, 24).to(device=cfg.device)
        self.model = EnsembleModel(model_factory, loss_factory, cfg.dynamics_model.ensemble_size, cfg.device)
        # NOT THREAD SAFE
        self.inputs = [None] * cfg.dynamics_model.ensemble_size
        self.outputs = [None] * cfg.dynamics_model.ensemble_size
        self.models = [None] * cfg.dynamics_model.ensemble_size
        self.workers = []
        for index in range(cfg.dynamics_model.ensemble_size):
            worker = BenchForwardMT2.ForwardWorker(index,
                                                   self.models,
                                                   self.inputs,
                                                   self.outputs,
                                                   self.input_cond,
                                                   self.output_events[index])
            worker.start()
            self.workers.append(worker)

    def __del__(self):
        for worker in self.workers:
            worker.stop()

        with self.input_cond:
            self.input_cond.notify_all()

        for worker in self.workers:
            worker.join()

    def run_benchmark(self, n):
        for i in range(n):
            def forward(models, inputs):
                # print("Forward called")
                assert len(models) == len(models)
                # put inputs in a well known place
                for idx, inp in enumerate(inputs):
                    self.inputs[idx] = inp
                    self.models[idx] = models[idx]

                with self.input_cond:
                    self.input_cond.notify_all()
                for idx in range(len(models)):
                    # print(f"Master waiting #{idx}")
                    self.output_events[idx].wait()
                    # print(f"Master woke #{idx}")
                    self.output_events[idx].wait()
                # print("Master finished waiting")
                return self.outputs

            self.model.forward(self.input1, forward)


class BenchComputeTrajectories(Benchmark):
    def __init__(self, jit=False):
        cfg = OmegaConf.load("benchmark/compute_trajectories.yaml")
        cfg.dynamics_model.jit = jit
        self.dynamics_model = NNBasedDynamicsModel(cfg, cfg.dynamics_model)

        env = gym.make(cfg.env.name)
        self.state0 = torch.from_numpy(env.reset()).to(dtype=torch.float32, device=cfg.device)
        action_size = np.sum(env.action_space.shape)
        actions = np.random.uniform(env.action_space.low, env.action_space.high,
                                    (num_trajectories, planning_horizon, action_size)).astype(dtype=np.float32)
        self.actions = torch.from_numpy(actions).to(device=cfg.device)

    def run_benchmark(self, n):
        for i in range(n):
            optimizers.Optimizer.compute_trajectories(self.dynamics_model, self.state0, self.actions)


class BenchmarkPlanActions(Benchmark):
    def __init__(self, jit=False):
        cfg = OmegaConf.load("benchmark/compute_trajectories.yaml")
        cfg.dynamics_model.jit = jit
        self.dynamics_model = NNBasedDynamicsModel(cfg, cfg.dynamics_model)
        env = gym.make(cfg.env.name)
        self.state0 = torch.from_numpy(env.reset()).to(dtype=torch.float32, device=cfg.device)
        self.opt = optimizers.RandomShootingOptimizer(device=cfg.device,
                                                      model=self.dynamics_model,
                                                      action_space=env.action_space,
                                                      env_config=cfg.env,
                                                      planning_horizon=planning_horizon,
                                                      num_trajectories=num_trajectories,
                                                      num_particles=num_particles)
        action_size = np.sum(env.action_space.shape)
        actions = np.random.uniform(env.action_space.low, env.action_space.high,
                                    (num_trajectories, planning_horizon, action_size)).astype(dtype=np.float32)
        self.actions = torch.from_numpy(actions).to(device=cfg.device)

    def run_benchmark(self, n):
        for i in range(n):
            a = self.opt.plan_action_sequence(self.state0)


class BenchmarkEnsemble(Benchmark):
    def __init__(self):
        cfg = OmegaConf.load("benchmark/compute_trajectories.yaml")
        type_str = cfg.dynamics_model.type.upper()
        params = cfg.dynamics_model[type_str]

        def model_factory():
            return utils.instantiate(params, 24, 16, cfg.device)

        def loss_factory():
            return utils.instantiate(params.loss)

        self.input1 = torch.zeros(num_trajectories, 24).to(device=cfg.device)
        self.model = EnsembleModel(model_factory, loss_factory, cfg.dynamics_model.ensemble_size, cfg.device)

        self.traced = torch.jit.trace(self.model, self.input1)
        # jit the model
        self.traced(self.input1)

    def run_benchmark(self, n):
        for i in range(n):
            self.traced(self.input1)  # use the jitted model
            # self.model.forward(self.input1)  # use the original model


def bench(b, n):
    if n == 0:
        return
    t = time()
    b.run_benchmark(n)
    torch.cuda.synchronize()
    e = 1000 * (time() - t)
    print(f"Elapsed {e:.3f} ms ({(e / n):.3f} ms per iteration)")


if __name__ == '__main__':
    # Single forward of the ensemble
    # b = BenchForward()
    # b = BenchForwardMT2()
    # b = BenchmarkEnsemble()

    # Computing num_trajectories of planning_horizon length
    # b = BenchComputeTrajectories(jit=False)
    b = BenchmarkPlanActions(jit=True)

    # warmup
    bench(b, 1)
    print("starting...")
    prof_start()
    bench(b, 100)
    prof_stop()
    print("done")

