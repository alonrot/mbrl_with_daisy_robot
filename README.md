Introduction
============
This branch of `mbrl` is dedicated to conducting experiments using [PETS](https://arxiv.org/pdf/1805.12114.pdf) on the real robot [Daisy](https://fb.quip.com/acOaAgiGoMWR). It supports additional changes with respect to the main branch, which include
 + Alternative trajectory sampling methods: `PDDM`([paper](https://arxiv.org/pdf/1909.11652.pdf)) and a new idea `Directed Brownian Motion`

![daisy](https://github.com/fair-robotics/mbrl/blob/hydra_with_daisy/pics/daisy_pic.png)

This project is work in progress.

Project features
----------------
 + Flexible framework for conducting experiments using PETS on the real robot Daisy
 + Reliable and empirically tested interface PETS <-> Daisy
 + Commented code with thorough explanations


Installation
============
> TODOL Make sure both mbrl and control-daisy are in the same folder


Infrastructure for interfacing with Daisy
=========================================
Depending on the "amount of data" we send to Daisy through the network, we may communicate PETS with Daisy in different ways. 
1. **Network interface**: The Hebi API runs as a main program in one terminal, and PETS runs as another main program in another terminal. Both programs share data through the network using UDP socket communication. Herein, we differentiate three situations, depending on *where* those terminals are running:
    1. The Hebi API runs in Daisy's onboard computer and PETs runs on the robodev
    2. Both run on the robodev
    3. Both run on Daisy's on-board computer
2. **Direct interface**: Both, the Hebi API is called during PETS initialization and thus, they both run as a single main program.

For doing learning experiments, the **Network interface** is more appropriate as we have decoupled Daisy's program from PETS. In this way, if there is a problem with the robot, we can kill the Habi API without having to restart the learning experiments from scratch (i.e., without having to kill PETS). We'd discourage anyone from using the **Direct interface** for doing learning experiments, as both PETS and the Hebi API are running in the same main program. Thus, having a problem in the robot would mean that both, PETS and the interface need to be killed at once. However, the **Direct interface** it's also supported and working, and can be used for testing/debugging purposes, as it doesn't rely on socket communication and thus there's no data being sent between the Hebi API and PETS.

In the following, we'll describe how to use the **Network interface** for the case 1.1. The following picture describes in a high level the communication between the machine where PETS is running and Daisy's on-board computer during a single time step. 
![env_network](https://github.com/fair-robotics/mbrl/blob/hydra_with_daisy/pics/env_network.png)
Importantly, PETS runs multiple episodes that last for N time steps each. At each time step, PETS needs to send an action to the robot (a_t), and retrieve the next state (x_{t+1}). As shown in the diagram, this is done using publishers/listeners running at different frequencies. Such publishers/listener send/receive data using UDP socket communication. We describe next the main three communication channels that take care of sending the needed data.

 + The middle channel sends the desired action to Daisy. At the moment, actions a_t are sent at 10 Hz. Since we are controlling the robot using a position controller, we choose such action to be directly the desired joint positions (q_{des}). On Daisy's on-board computer, a listener is running at 100 Hz. The purpose of such listener is two fold: First, capturing actions sent from PETS, and second, updating a "position holder". The position holder is simply a while-loop that reads the desired actions as they arrive and passes them to the Hebi API.

 + The bottom channel collects observations from the Hebi API (x_{t+1}) and publishes them at 100Hz. Then, in PETS, x_{t+1} is received at 10Hz, and used to continue to the next step.

 + The top channel sends relevant information about Daisy to PETS encoded as an integer. Negative integers correspond to `error codes`, such as, one of the motors having a temperature above the critical threshold, or the vision device having lost track. PETS needs to be aware of these `bad robot status` to immediately stop the episode, and let the user decide about how to follow up, e.g., restart the episode and discard the data, continue anyways and use/not use the collected data for training the model, etc.

The cases 1.2 can be achieved with the same interface but changing the network address to ("localhost",PORT). We'll describe it later on.

> This infrastructure has proven to work reliably adding only small delays to the existing delays in the WiFi network.

Running the **Network interface** for conducting learning experiments
=====================================================================
Running learning experiments involves two major steps: First, bringing up the communication in Daisy's on-board computer. Second, starting PETS, and with it, the learning experiments.

> TODO: Make sure we point to control-daisy for how to start the robot.

Start communication in Daisy's on-board computer
------------------------------------------------
 + Get Daisy ready to go by following the next instructions.
 + Connect the robodev to the `hebirobot` WiFi.
    + Important: In my experience, the WiFi card integrated in the BIOS of the robodev didn't work reliably (for example, I couldn't ssh into Daisy's computer). You can use external WiFi devices you can plug with USB, however, those aren't great either.
    + You may have a few problems with the external WiFi devices. If it doesn't connect less than 10 seconds, unplug and replug.
 + Open a new terminal and ssh to Daisy's computer and go to the path where the repo is stored

    ```bash
    ssh hebi@10.10.1.2
    cd <path/to/control-daisy>
    ```
 + Set up your communication variables by editing the file control-daisy/conf/communication_with_PETS.yaml
   Make sure they look like this

    ```yaml
    listen_or_publish: listen

    listener:
      communication:
        UDP_IP: "10.10.1.184" # IP address of the robodev
        HOST: "10.10.1.2" # IP address of Daisy's on-board computer

      daisy:
        init:
          freq_fbk: 1000 # Hz
          use_fake_robot_debug: False # Set this to True if you don't want to use the robot (e.g., for debugging)
          use_fake_vision_debug: False # Set this to True if you don't want to use the robot (e.g., for debugging)
        junk_matrix_vision_length: 50

        hold_position_process:
          freq: 200 # Hz
          name_proc: "daisy_position_holder_listener"
        current_position_process_publisher:
          freq: 100 # Hz
          name_proc: "daisy_current_state_socket_publisher"
        status_robot_process_publisher:
          freq: 5.0 # Hz
          name_proc: "daisy_is_alive_socket_publisher"
        desired_position_process_listener:
          name_proc: "daisy_desired_position_socket_listener"
          timeout_socket: 100000. # [sec] timeout to close the socket, counted from the last message arrived

    common:
      PORT_DES_POSITIONS: 50012
      PORT_CURR_POSITIONS: 50013
      PORT_ACK: 50011
      PORT_FLAG_RESET: 50010
      PORT_STATUS_ROBOT: 50009
    ```

    > Although we specify "listen_or_publish: listen" this doesn't mean that the program *only* listens. As shown in the diagram, the program also publishes a state vector (bottom channel) and a robot status (top channel).
 + Run the following command
    ```
    python run_communication_with_PETS.py
    ```
 + First, the robot and the vision device will be searched in the network.
 + If all went well, all the publishers/listeners will be started and the program will be on hold, essentially, ready for receiving commands from PETS. A sign of things going well is that you should hear a *buzz* in Daisy, indicating that the motors are receiving a "position to hold", i.e., the first position to be read.

A few notes about this:
> The IP `10.10.1.2` is fixed, and the password is `hebi1234`.

> The parameter "UDP_IP" is the IP of the computer we want to send data to, in this case, the robodev. You can see your current IP address by clicking on the WiFi icon in the top toolbar, and the click on "Connection information".

> Make sure you don't change the port numbers.

> The parameters "use_fake_robot_debug" and "use_fake_vision_debug" allow you to debug your code without using the robot. For this, set both flags to True. If False, the program will look for the robot modules (i.e., the 18 motors) at initialization, and also the smartphone. When any of those are not found, the program will prematurely terminate.

Start PETS
----------
Depending on the nature of the learning experiments, you may specify different settings in the corresponding yaml files, to run PETS in one way or another. For example, you might want to only collect data with a specific policy without training a dynamics model, or pre-load an existing model and run 100 episodes of 300 steps each, retraining the model only every 500 datapoints, and not plotting the reward evolution. All these options are decoupled and described later on.
 + Open a new terminal and cd to the path where you have mbrl
    ```bash
    cd <path/to/mbrl>
    ```
 + Make sure your pets/conf/config.yaml contains the following basic setup
    ```yaml
    defaults:
     - env: ???
     - dynamics_model: pe
     - optimizer: pddm                       # When changing this, also policy.clazz should change in daisy.yaml to clazz: pets.PETSPolicy
     - env/: ${defaults.0.env}/dynamics_model/${defaults.1.dynamics_model}
     - env/: ${defaults.0.env}/optimizer/${defaults.2.optimizer}
     - traj: ts1
     - hydra/launcher: mbrl_fairtask
     - ./: ../../plotting_and_analysis/conf_data2load
     - schema: debug
    ```
    The symbol `???` means that the value is mandatory and should be resolved either from another yaml file, or passing it in the command line.
    If using `env: daisy_real`, the file pets/conf/env/daisy_real.yaml will be loaded.
    The `schema` is used to overwrite any values pacified in such file and in all the others.
    The `schema: debug` can be used to debug PETS without prompting the user.
    The file ./: ../../plotting_and_analysis/conf_data2load.yaml is THE file where all the collected data/trained models are labeled 
    for running PETS and also for plotting. If a model is to be pre-loaded for running PETS, such file should be used.

 + The file pets/conf/env/daisy_real.yaml is fully commented and should be self-explanatory. Make sure that the following minimal
 configuration is present.

    ```yaml

    # ============================
    # Environment specific options
    # ============================
    env:
      which_interface: "network"
      interface:
        network:
          HOST: "10.10.1.180" # alonrot robodev (cable)
          UDP_IP: "10.10.1.2"
          PORT_DES_POSITIONS: 50012
          PORT_CURR_POSITIONS: 50013
          PORT_ACK: 50011
          PORT_FLAG_RESET: 50010
          PORT_STATUS_ROBOT: 50009

      name: DaisyRealRobotEnv
      freq_action: 10.0 # [Hz] # Frequency at which actions are sent from PETS

    # ===========
    # PETS policy
    # ===========
    policy:
      clazz: pets.PETSPolicy              # When changing this, also the cem should change to cem_parametrized in config.yaml
      params:
        planning_horizon: 5 # Planning horizon of the MPC optimizer. The longest, the longer will each step take to finish.

    # ======================
    # Model training options
    # ======================
    training:
      do_train_over_episodes: True # If set to False, training will be skipped for all the episodes
      batch_size: 60 # Train the NN in batches of this size
      full_epochs: 2000 # Number of epochs
      incremental_epochs: 120 # Number epochs to train the model with while PETS is running (i.e., this is not for pre-training the model)
      optimizer:
        params:
          lr: 1e-5 # Learning rate

    # Reward class
    # ============
    reward:
      clazz: mbrl.rewards.RewardWalkForward
      params:
        dim_state: 45

    ```
> Note that the PORT numbers have to coincide with those from control-daisy/conf/communication_with_PETS.yaml. Also, note that the HOST/UDP_IP are flipped with respect to those in control-daisy/conf/communication_with_PETS.yaml. This makes sense, as in the robodev, the host it the robodev itself (*IP: 10.10.1.180*) and PETS wants to send data to Daisy using its *IP: 10.10.1.2* as identifier.

> If you want to run both PETS and the Daisy API in the same computer but still rely on the socket communication, you can simply set
    ```yaml
        HOST: "localhost"
        UDP_IP: "localhost"
    ```
    both in pets/conf/env/daisy_real.yaml and in control-daisy/conf/communication_with_PETS.yaml. This is very useful for debugging purposes, i.e., debugging your code without having to connect with the real robot nor use Daisy's onboard computer.

> Make sure that training.do_train_over_episodes is set to True if you want to retrain the model as new data is collected from episodes.

> Make sure the reward function is the one you want. We have created a new module `mbrl.rewards` to store reward functions, instead of having static rewards functions as in the main branch. This is useful as some parameters (e.g., the goal) are known a priori and can be preallocated. Since the reward function is called many times, this can make a difference in performance.

> freq_action has to be such that the function env.step() inside mbrl.utils.sample_episode() executes below 1./freq_action [sec]. In other words, we can't send actions faster than PETS finishes to execute one step. If this happens, a message THE LOOP IS SATURATED will be printed in the screen. There's no way to know a priori it this will be the case. To test this, run PETS `schema: debug` (no need to start the real robot for this) and make sure that the loop is not saturated.

 + Run PETS
    ```
    python pets/main.py env=daisy_real device=cuda
    ```
If all is running correctly, the robot should start moving and you should see prints in both terminals.

Notes
-----
> Currently, only env=daisy_real is supported. In old commits, running PETS on simulated Daisy was supported. For an example, I'd recommend checking out the following specific commit and running pets using the simulated environment (i.e., daisy)
```bash
git checkout b821e9a
python pets/main.py env=daisy
```

