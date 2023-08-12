# mbrl/pets

Work in progress

## Setting Up
[Setup instructions for robotics lab and FAIR cluster](docs/README.md)

## Bash Completion
Run this for awesome bash TAB completion. See [this](bash_completion/) for more detail.
```bash
source pets/bash_completion/main_py.bash
```

## Running
The mbrl codebase is transitioning to use Hydra (https://github.com/fairinternal/hydra), so there is some depreciated 
code for legacy support (AbstractMain, Launchers).
Run from the top level repo directory.

env takes an environment name, currently cartpole and halfcheetah are supported.
To run cartpole:
```
python pets/main.py env=cartpole random_seed=1 hydra/launcher=mbrl_fairtask 
```
(`hydra/launcher=mbrl_fairtask` is currently not used when not running locally, will push a fix to not need it soon)
For local runs, required values are:
- env: The gym env to run
- random_seed: single value, or sweep over multiple (see below)

Any part of the configuration can be overridden from the command line, for example, batch size can be overriden with
```
python pets/main.py  env=cartpole random_seed=1 training.batch_size=64
``` 

### Defaults
Presets are a way to specify a collection of configuration options by name.
supported defaults (preset in bold):

optimizer:
  * random: random shooter optimizer
  * **cem**: cross entropy method optimizer

dynamics_model:
  * d : deterministic neural network
  * **p** : Probabilistic neural network
  * de : Ensemble of deterministic neural networks
  * pe : Ensemble of Probabilistic neural networks
  * env : Using environment as the dynamics model


Multiple defaults can be used at once, for example - 
To run with deterministic dynamics model and random shooter optimizer:
```
python pets/main.py env=cartpole dynamics_model=d optimizer=random random_seed=1
```

## Distributed launches
fairtask is used to submit multiple jobs to slurm or locally. Hydra also includes support for 
submitit.
To launch a job, the Hydra configuration for multirun ```-m --multirun``` is used. (Note old documentation may reference `-s` for sweep).
All variables must come after `-m`.
Currently, launching the same job with multiple random seed is supported.
Note that [fairtask-slurm](https://github.com/fairinternal/fairtask-slurm) needs to be install for this to work.

```
# To launch a job with the default options:
python pets/main.py -m  env=cartpole random_seed=1 partition=learnfair hydra/launcher=mbrl_fairtask 

# To launch 5 instances, overriding number of trials per run to 1:
python pets/main.py -m  env=cartpole random_seed=1,2,3,4,5 partition=learnfair hydra/launcher=mbrl_fairtask num_trials=1 
```

Required parameters for launching jobs:
- hydra/launcher: A hydra launcher config modified for more flexibility in the mbrl framework. Other options then those in `pets/conf/hydra/launcher` include fairtask, submitit.
- partition: Normal use is learnfair, or priority (for within 2 weeks of a deadline). For more [read here](https://our.internmc.facebook.com/intern/wiki/FAIR/Platforms/FAIRClusters/SLURMGuide/#partitions-on-the-cluste).
- random_seed: This will control the number of parallel jobs you run. (e.g. random_seed=1 will launch one of each, and random_seed=1,2,3,4 will launch for jobs for each of the other variables. This is the most basic parameter sweep)

## Parameter sweeps
Parameter sweeps are supported with the -s *key=v1,v2,v3* syntax.
Both presets and regular parameters can participate in the sweep.
If multiple lists are specified, the cartesian product will be launched.
for example, the following command will run 4 configurations:

```
python pets/main.py -m  env=cartpole random_seed=1,2,3 optimizer=cem,random  partition=priority hydra/launcher=mbrl_fairtask
[2019-09-27 13:44:10,031] - Sweep output dir : /checkpoint/nol/outputs/2019-09-27/13-44-10
[2019-09-27 13:44:10,083] - 	#0 : env=cartpole random_seed=1 optimizer=cem partition=learnfair
[2019-09-27 13:44:10,083] - 	#1 : env=cartpole random_seed=1 optimizer=random partition=learnfair
[2019-09-27 13:44:10,083] - 	#2 : env=cartpole random_seed=2 optimizer=cem partition=learnfair
[2019-09-27 13:44:10,083] - 	#3 : env=cartpole random_seed=2 optimizer=random partition=learnfair
[2019-09-27 13:44:10,083] - 	#4 : env=cartpole random_seed=3 optimizer=cem partition=learnfair
[2019-09-27 13:44:10,083] - 	#5 : env=cartpole random_seed=3 optimizer=random partition=learnfair
```

| defaults.optimizer   | random_seed |
| ------------- |:-------------:|
| cem           | 1            |
| random        | 1            |
| cem           | 2            |
| random        | 2            |
| cem           | 3            |
| random        | 3            |

Note that by default 1 jobs of each configuration with default random seed is launched. In order to create parallel
of each job, sweep across random seed as well. 


## Visualizing
Visualization is still being finalized for ease of use in the Hydra implementation.
Visualization is done using [Visdom](https://github.com/facebookresearch/visdom).
Run a local Visdom server and connect with your browser.

Example:
Note that visdom environment env name allow you to easily compare different experiments.

```
python pets/visualize/visualize_pets.py visualize_group -d log_dir1 log_dir2 ...
```

To visualize a sweep top level directory:
```
python pets/visualize/visualize_pets.py visualize_sweep -d sweep_dir
```


## Logging
Logging is still being finalized for ease of use in the Hydra implementation.
**Hydra handles logging by default**.
Logging is configured by conf/logging.yaml, but specific loggers can be turned to debug from the command line:
For example the following activate debug modules to all the files under optimizers and policies
```
python pets/main.py -v mbrl.optimizers,mbrl.policies train -e cartpole

```
Log files are stored under the log directory for the job, for example `logs/2019-01-15_17-42-34/log.log`

## Debug Tools:
- `-v 'hydra|root'` prints all actions performed by Hydra when initializing your job, use this if you do not know why a certain cfg value is changed.
- `-c` prints the config but does not run the experiment. Currenly conflicts with `-m`.
