# h1v2 Isaac

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-BSD%202--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

## Dependencies

## Overview

This repository provides the essential codebase to run custom RL policies on an [Unitree H1-2](https://support.unitree.com/home/en/H1_developer/About_H1-2) robot.

Currently the RL training uses [Isaac Lab](https://github.com/isaac-sim/IsaacLab).
The sim2sim validation is done with [Mujoco](https://github.com/google-deepmind/mujoco).
The sim2real deployment pipe depends on [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python).

The project is structured around three main packages:

- **robot_assets**: Manages the models of the robot
- **robot_deploy**: Handles deployment-related functionalities (sim2sim and sim2real).
- **robot_tasks**: Contains the core definition of the training environment for the robot.

Both `robot_deploy` and `robot_assets` are adapted from the template provided for [Isaac Lab Projects](https://github.com/isaac-sim/IsaacLabExtensionTemplate) to facilitate integration.

## Usage

This project has been packaged using [UV](https://docs.astral.sh/uv/).

To run a training:

```bash
uv run scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-H12_12dof-v0
```

This command initiates the training process for the specified task.

To deploy a trained policy onto a real robot, use

```bash
uv run --package robot_deploy scripts/deploy/main.py
```

**Note:** the `--package robot_deploy` flag avoids pulling the dependencies from the `robot_tasks` package used for training (e.g. `isaacsim`, `isaaclab`...).

To evaluate a policy in simulation, set the `use_mujoco` flag to True in the `config.yaml` file before running the `main.py` script: it will spawn a MuJoCo simulator instance in the background and communicate with it through DDS, as when deploying on a real robot.

It's also possible to run the trained policy in simulation directly (i.e. without using DDS) by passing the flag `--sim` to the `main.py` script, in this case the policy is synchronously run in the simulator.

## Input devices

Currently, three input methods are implemented to control the robot's behavior:

- **Unitree controller**: when the code is not run in simulation (i.e. no flag `--sim` AND `use_mujoco: False` in the configuration file), the default input device is the Unitree Controller
- **Gamepad**: when running in simulation, the code will automatically attempt to detect and use any connected gamepad
- **Keyboard**: if no gamepad is detected, keyboard inputs are read  through the MuJoCo simulator window

The default keybindings are:

- `start`: initialize the robot
- `select`: kill the robot and activate damping mode
- `L1` / `R1`: switch between control policies specified in the configuration file
- *(in MuJoCo only)* `B`: toggle the elastic band maintaining the robot in its standing position
- *(in MuJoCo only)* `L2` / `R2`: modify the length of the elastic band

The keyboard inputs are mapped to the generic controller commands as follows:

| Keyboard key | Mapped key |
|:------------:|:----------:|
| Enter        | Start      |
| Escape       | Select     |
| A/B/X/Y      | A/B/X/Y    |
| J            | L1         |
| K            | R1         |
| I            | L2         |
| O            | R2         |

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details.

## Contributors

- [Valentin Guillet](https://github.com/Valentin-Guillet): Core developer
- [Côme Perrot](https://github.com/ComePerrot): Core developer
- [Constant Roux](https://github.com/ConstantRoux): Core developer
- [Victor Lutz](https://github.com/vicltz): Robot model integration
- [Alessandro Trovatello](https://github.com/alessandrotrovatello): RSL-RL training implementation
- [Olivier Stasse](https://github.com/olivier-stasse): Project supervisor

## Citation

To cite this work in a publication:

```text
@misc{h1v2Isaac2025,
  author = {Valentin Guillet and Côme Perrot and Constant Roux and Olivier Stasse},
  title = {h1v2-Isaac: Reinforcement Learning Framework for Unitree H1-2 Robot},
  year = {2025},
  howpublished = {\url{https://github.com/Gepetto/h1v2-Isaac}},
}
```
