# FAIR cluster setup

Follow the instructions in [Robotics lab setup](robotics_lab.md) to install MuJoCo.

## Remote PyCharm
* Create a project from github on laptop
* In Preferences->Deployment, create a connection to your devfair via an ssh tunnel (See [Connection](Deployment_Connection.png) and [mapping](Deployment_Mapping.png) tabs)
* Sync to the devfair Tools->Deployment->Sync with, after the first time you can just upload instead.
* You can set up automatic upload from *Preferences->Build,Exectution,Deployment->Deployment->Options*

Once the directory is uploaded, on the devfair:

## devfair setup
Either checkout from github directly or follow the Remote Pycharm section.
activate conda module and create the environment:
```
~/dev/mbrl/pets$ module load anaconda3/5.0.1
~/dev/mbrl/pets$ conda env create -f environment.yml
```
*Note: on the devfair, you may see an error installing mujoco-py after which the installer will automatically
install the missing dependencies. this is normal on the FAIR cluster and is probably related to the old version
of Anaconda there (This problem does not exist in the Robotics lab)*

Activate environment with `source activate pets`

You can test that it's working by running `python mbrl.py train -e=MBRLCartpole-v0`, see [README.md](../README.md) for full details

## Adding SSH Python interpreter in PyCharm
If you are setting up Remote PyCharm, adding a remote SSH interpreter will allow you to run and debug directly
on the FAIR cluster from your laptop.

Once you have the conda environment setup on your devfair, you can tell PyCharm to use it as your project interpreter.

* Preferences -> Project: -> Project Interpreter
* SSH Interpreter -> Use existing -> Select your connection
* For Interpreter, enter the output of `which python` on the devfair while the
environment is active, in my case `/private/home/omry/.conda/envs/pets/bin/python`

PyCharm is an idiot, so it messes up your mappings in the connection you created.
Go back and fix it based on the [mapping](Deployment_Mapping.png) photo.




