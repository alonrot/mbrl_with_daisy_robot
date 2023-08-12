import logging
import sys

import numpy as np
import torch


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def config_log():
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
