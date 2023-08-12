print("setting up profiler")
import ctypes

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

print("loading pytorch")
import torch

print("setting up benchmark")
k = 5
w = 1000
tensors = [torch.ones(w, w).cuda() for _ in range(k)]
output = torch.zeros(w, w).cuda()
streams = [torch.cuda.Stream() for _ in range(k)]

print("starting...")
prof_start()

for idx in range(k):
    with torch.cuda.stream(streams[idx]):
        torch.matmul(tensors[idx], tensors[idx], out=output)

for idx in range(k):
    torch.cuda.current_stream().wait_stream(streams[idx])

prof_stop()
print("done")
