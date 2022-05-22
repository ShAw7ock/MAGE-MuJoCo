Model-based Action-Gradient-Estimator (MAGE)
===================

This is a simple implementation for [MAGE](https://arxiv.org/pdf/2004.14309.pdf) algorithm.
The author's code can be seen [here](https://github.com/nnaisense/MAGE).

## NOTE:
* Wrapped MuJoCo environments bound the action spaces in the scale of `[-1, 1]`.
* Run the code with: `python ./main.py --env_name ENV_NAME` which can be seen in: `./envs/gymmb/__init__.py`.
* The process of policy optimization is based on [TD3](http://proceedings.mlr.press/v80/fujimoto18a/fujimoto18a.pdf): `./algo/td3.py`
* Modify the Hyper-parameters in: `./components/arguments.py`
* World Models can be seen in: `./components/dynamics.py`

## Requirements
* Python >= 3.6.0
* PyTorch == 1.7.0 (optional)
* [MuJoCo 200](https://roboti.us/)
* [mujoco-py](https://github.com/openai/mujoco-py) == 2.0.2.8
* OpenAI Gym == 0.17.0

# Acknowledgement
This code is referenced by [nnaisense/MAGE](https://github.com/nnaisense/MAGE). <br>