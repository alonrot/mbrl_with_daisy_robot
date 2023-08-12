import os
import sys

# add cwd to path to allow running directly from the repo top level directory
sys.path.append(os.getcwd())

from omegaconf import OmegaConf
from pets.bash_completion.suggest import get_suggestions

CONFIG=f"""
random_seed: 100
training:
  batch_size: 16
  full_epochs: 100
  incremental: false
  incremental_epochs: 5
  optimizer:
    clazz: torch.optim.Adam
    params:
      lr: 0.0001
  testing:
    split: 0.9
trial_timesteps: 200
"""

def test_bash_completion():
    config = OmegaConf.from_string(CONFIG)

    level_1_suggestions = ['random_seed=', 'training.', 'trial_timesteps=']
    assert level_1_suggestions == get_suggestions(config, "")
    assert level_1_suggestions == get_suggestions(config, "tra")
    assert level_1_suggestions == get_suggestions(config, "training")

    level_2_suggestions = ['training.batch_size=', 'training.full_epochs=', 'training.incremental=', 'training.incremental_epochs=', 'training.optimizer.', 'training.testing.']
    assert level_2_suggestions == get_suggestions(config, "training.")
    assert level_2_suggestions == get_suggestions(config, "training.batch")

    level_3_suggestions = ['training.optimizer.clazz=', 'training.optimizer.params.']
    assert level_3_suggestions == get_suggestions(config, "training.optimizer.")
    assert level_3_suggestions == get_suggestions(config, "training.optimizer.par")
    assert level_3_suggestions == get_suggestions(config, "training.optimizer.params")

    level_4_suggestions = ['training.optimizer.params.lr=']
    assert level_4_suggestions == get_suggestions(config, "training.optimizer.params.")

    level_5_suggestions = ['0.0001']
    assert level_5_suggestions == get_suggestions(config, cur_word="=", prev_word="training.optimizer.params.lr")
