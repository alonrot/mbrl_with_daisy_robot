# Robotics lab setup
- Download MuJoCo 1.50 from https://www.roboti.us/index.html, unzip it into `~/.mujoco/mjpro150`, and place your license key (the `mjkey.txt` file) at `~/.mujoco/mjkey.txt`

```bash
wget https://www.roboti.us/download/mjpro150_linux.zip
mkdir ~/.mujoco/
unzip mjpro150_linux.zip -d ~/.mujoco/
mv <path_to_mjkey.txt> ~/.mujoco/mjkey.txt
```

- Copy `envrc.lab.sample` to `.envrc` and allow direnv to activate it when you enter the directory for the first time with `direnv allow`.
Direnv will create a new conda environment called *pets* and will activate it automatically when you get into the mbrl directory.

```bash
# Inside the mbrl repository
cp envrc.lab.sample .envrc
```

At this point you can use local or remote Pycharm to develop the project, use the created *pets* conda environment as your python interpreter.
