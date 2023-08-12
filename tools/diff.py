#!/usr/bin/env python3
import os
import sys

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())

import re
import dill
import torch
import numpy as np
import numbers
import argparse
from fnmatch import fnmatch

def get_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('file1', type=str, help='file1')
    parser.add_argument('file2', type=str, help='file2')
    parser.add_argument('--ignore-class', type=str, help='list of class to ignore. Ex: mtrand.RandomState, *.RandomState, etc.',
                        default='mtrand.RandomState')
    parser.add_argument('--ignore-key', type=str, help='list of keys to ignore. Ex: *time, or *rng_state, etc.',
                        default='time,total_time,torch_cuda_rng_state')

    args = parser.parse_args()

    args.whitelist_keys = args.ignore_key.split(',')

    default_whitelist_class = ['mtrand.RandomState']
    args.whitelist_classes = set(args.ignore_class.split(',') + default_whitelist_class)

    return args

def fullname(obj):
  module = obj.__class__.__module__
  if module is None or module == str.__class__.__module__:
    return obj.__class__.__name__  # Avoid reporting __builtin__
  else:
    return module + '.' + obj.__class__.__name__

def is_equal(a, b, whitelist_classes=[], whitelist_keys=[]):

    diffs = []

    def decorator(func):
        def wrapper(a, b, path=""):
            e = func(a, b, path)
            if not e:
                diff = {'path': path, 'class': type(a)}
                diffs.append(diff)
            return e
        return wrapper

    @decorator
    def equal(a, b, path=""):

        if type(a) != type(b):
            return False

        for cls in whitelist_classes:
            if fnmatch(fullname(a), cls):
                return True

        if isinstance(a, np.ndarray):
            return np.all(a == b)

        if isinstance(a, torch.Tensor):
            return torch.equal(a, b)

        # List or Tuple
        if isinstance(a, list) or isinstance(a, tuple):
            if len(a) != len(b):
                return False

            return np.all([
                equal(aa, bb, f"{path}[{i}]")
                for i, (aa, bb) in enumerate(zip(a, b))
            ])

        # Dictionary
        if hasattr(a, 'keys'):
            if set(a.keys()) != set(b.keys()):
                return False

            return np.all([
                equal(a[k], b[k], f"{path}.{k}") for k in a.keys()
                if k not in whitelist_keys
            ])

        return a == b

    e = equal(a, b)

    return e, diffs

def load_data(filenames):

    file_ext = os.path.splitext(filenames[0])[1]
    if file_ext == ".dat":
        file_loader = lambda x: torch.load(x)
    elif file_ext == ".pkl":
        file_loader = lambda x: dill.load(open(x, 'rb'))

    data = [file_loader(filename) for filename in filenames]

    return data

def report_diffs_and_exit(diffs):
    for diff in diffs:
        sys.stderr.write("\33[33m{}\33[0m [{}]\n".format(
            diff['path'], diff['class']
        ))
    sys.exit(2)

def main():

    args = get_args()

    # TODO(poweic): support more files at the same time
    filenames = [args.file1, args.file2]

    data = load_data(filenames)

    same, diffs = is_equal(data[0], data[1],
                           whitelist_classes = args.whitelist_classes,
                           whitelist_keys = args.whitelist_keys)

    msg_prefix = f"Binary files \33[32m{filenames[0]}\33[0m and \33[32m{filenames[1]}\33[0m"
    if not same:
        sys.stderr.write(f"{msg_prefix} differ in:\n")
        report_diffs_and_exit(diffs)

    print (f"{msg_prefix} only differ in keys that are whitelisted:")
    print (f"{args.whitelist_keys}")

if __name__ == "__main__":
    main()
