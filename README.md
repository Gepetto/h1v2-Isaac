# h1v2 Isaac

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.1.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-BSD%202--Clause-blue.svg)](https://opensource.org/licenses/BSD-2-Clause)

## Overview

This repository provides the essential codebase to run custom RL policies on an [Unitree H1-2](https://support.unitree.com/home/en/H1_developer/About_H1-2) robot.

Currently the RL training uses [Isaac Lab](https://github.com/isaac-sim/IsaacLab).
The sim2sim validation is done with [Mujoco](https://github.com/google-deepmind/mujoco).
The sim2real deployment pipe depends on [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python).

The project is structured around three main packages:
- **biped_assets**: Manages the models of the robot
- **biped_deploy**: Handles deployment-related functionalities (sim2sim and sim2real).
- **biped_tasks**: Contains the core definition of the training environment for the robot.

Both `biped_deploy` and `biped_assets` are adapted from the template provided for [Isaac Lab Projects](https://github.com/isaac-sim/IsaacLabExtensionTemplate) to facilitate integration.

## Usage

This project has been packaged using [UV](https://docs.astral.sh/uv/).

To run a training:
```bash
uv run scripts/rsl_rl/train.py --task Isaac-Velocity-Flat-H12_12dof-v0
```
This command initiates the training process for the specified task.

For the sim2sim:
```bash
uv run scripts/deploy/sim2sim.py
```
This command runs the trained policy inside of MuJoCo.

For the sim2real:
```bash
uv run scripts/deploy/sim2real.py
```
This command can be used to deploy the trained policy onto a real robot.

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE) file for details.

## Contributors

- [Valentin Guillet](https://github.com/Valentin-Guillet)
- [Côme Perrot](https://github.com/ComePerrot)
- [Constant Roux](https://github.com/ConstantRoux)