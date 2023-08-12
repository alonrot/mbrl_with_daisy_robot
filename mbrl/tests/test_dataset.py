import os
import sys

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())

import pytest
import torch
import numpy as np
import tempfile
import mbrl.dataset

def test_appending_torch_tensor():
    """
    Randomly create a bunch of vectors with same shape and store them in
    TensorList. Compare data stored in TensorList to those in Python list.
    """
    dim = 5
    num_times_to_append = 7

    for shape in [[1], [dim], [1, dim], [dim, 1]]:
        tensors = mbrl.dataset.TensorList()
        tensors_in_list = []

        for _ in range(num_times_to_append):
            r = torch.randn(shape)
            tensors_in_list.append(r)
            tensors.append(r)

        tensors_in_list = torch.stack([t.reshape(-1) for t in tensors_in_list])

        # Test __len__
        assert len(tensors) == tensors_in_list.shape[0]

        # Test correctness of append & expand function
        assert torch.equal(tensors.data[:len(tensors)], tensors_in_list)

        # Test __getitem__
        for i in range(len(tensors)):
            assert torch.equal(tensors[i], tensors_in_list[i])

def test_appending_tensors_with_different_dimensions():
    """
    A list with tensors having different dimensions (ex: 1 x 5 and 1 x 7) is
    not allowed. An AssertionError should be raised.
    """
    dim1 = 5
    dim2 = 7

    with pytest.raises(AssertionError) as e_info:
        t = mbrl.dataset.TensorList()
        t.append(torch.ones(dim1))
        t.append(torch.ones(dim2))

def test_tensor_list_getitem():
    num_elems = 7
    dim = 3
    tensors = mbrl.dataset.TensorList()

    # create a list of 1, 2, 3 ... and append them to tensors
    test_values = [np.ones(dim) * i for i in range(num_elems)]
    for value in test_values:
        tensors.append(torch.Tensor(value))

    # index in the following should work as expected
    for i in range(-num_elems, num_elems):
        assert np.all(tensors[i].numpy() == test_values[i])

    # index in the following should raise IndexError (the same behavior as list)
    for i in [-num_elems - 2, -num_elems - 1, num_elems, num_elems + 1]:
        with pytest.raises(IndexError) as e_info:
            tensors[i]

def test_appending_tensors_with_different_shape_but_compatible():
    """
    Tensors like 1 x N and N x 1 have different shapes. Though they're
    compatible, we shouldn't ignore the difference and suppress the error
    silently. Same for 1 x N and N. An AssertionError should be raised.
    """
    from itertools import permutations

    dim = 5

    vec = torch.ones(dim)
    row_vec = torch.ones(1, dim)
    col_vec = torch.ones(dim, 1)

    # Test all combinations (with order, i.e. permutation)
    for v1, v2 in permutations([vec, row_vec, col_vec], 2):
        with pytest.raises(AssertionError) as e_info:
            t = mbrl.dataset.TensorList()
            t.append(v1)
            t.append(v2)

def test_sas_dataset():
    """
    Test SASDataset by creating and saving bunch of random states/actions in
    the SASDataset, and saving/loading to/from disk.
    """
    state_dim = 3
    action_dim = 1
    steps = 1000

    # Create a episode of random states and actions
    episode = [mbrl.dataset.SAS(
        s0=torch.randn(state_dim),
        a=torch.randn(action_dim),
        s1=torch.randn(state_dim)
    ) for _ in range(steps)]

    dataset = mbrl.dataset.SASDataset()

    # Test method add_episode
    dataset.add_episode(episode, device="cpu")

    # Test method __len__ and __getitem__ of class SASDataset
    # and method __eq__ of class SAS
    for i in range(len(dataset)):
        assert dataset[i] == episode[i]

    # Test iterating the dataset using `for ... in ...` syntax. Since we didn't
    # implement __iter__ method for SASDataset, the default behavior is to call
    # __getitem__ until an IndexError is raised. The expected behavior should
    # be the same as the above (just different syntax).
    for i, sas in enumerate(dataset):
        assert sas == episode[i]

    # Create a temporary file on the filesystem (will be auto removed)
    tmpfile = tempfile.NamedTemporaryFile()

    # Save the dataset
    torch.save(dataset, tmpfile.name)

    # Load it back
    dataset_restored = torch.load(tmpfile.name)

    # Test method __eq__ of class SASDataset
    assert dataset == dataset_restored

def sas_dataset_container_speed_comparison():
    """
    Compare the speed of serializing/deserializing and saving/loading data on
    disk with different containers. One is Python's built-in list and the other
    is the customized TensorList, which store vectors with same shape in a
    single huge tensor.
    """
    from time import time

    state_dim = 18
    action_dim = 6
    steps = int(3e5)

    repeat_n_times = 5

    print ("Profiling saving/loading dataset of {} tensors for {} times ..."
           .format(steps, repeat_n_times))

    # Create a episode of random states and actions
    for container in [list, mbrl.dataset.TensorList]:
        dataset = mbrl.dataset.SASDataset(container=container)
        for i in range(steps):
            dataset.add(mbrl.dataset.SAS(
                s0=torch.randn(state_dim, device='cpu'),
                a=torch.randn(action_dim, device='cpu'),
                s1=torch.randn(state_dim, device='cpu')
            ))

        t_save = 0
        t_load = 0
        for i in range(repeat_n_times):
            # Create a temporary file on the filesystem (will be auto removed)
            tmpfile = tempfile.NamedTemporaryFile()

            t_save -= time()
            torch.save(dataset, tmpfile.name)
            t_save += time()

            t_load -= time()
            _ = torch.load(tmpfile.name)
            t_load += time()

        avg_save_time = t_save / repeat_n_times
        avg_load_time = t_load / repeat_n_times
        print ("Took {:.5f} sec to save, {:.5f} sec to load [with container {}]"
               .format(avg_save_time, avg_load_time, container))

if __name__ == "__main__":
    sas_dataset_container_speed_comparison()
