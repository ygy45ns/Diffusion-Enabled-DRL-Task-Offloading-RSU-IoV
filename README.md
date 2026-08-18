# Multi-Hop Task Offloading Based on Diffusion-Enabled Deep Reinforcement Learning in RSU-Assisted Internet of Vehicles

This repository contains the experiment code for the paper *"Multi-Hop Task Offloading Based on Diffusion-Enabled Deep Reinforcement Learning in RSU-Assisted Internet of Vehicles"*.

We study multi-hop task offloading in RSU-assisted IoV, where RSUs can offload computation tasks to cloud servers, neighboring RSUs, and mobile vehicles (via multi-hop V2V relaying). A diffusion-enabled DRL method is proposed to make offloading decisions, and it is compared against DQN, A3C, and Greedy baselines.

## Repository Structure

```
env/                    # Simulation environment
├── config.py           # Environment parameters
├── datastruct.py       # Data structures (vehicles, RSUs, task queues, time slots)
├── environment.py      # Gym environment: state, action, reward, dynamics
└── utils.py            # Plotting and result-saving utilities

methods/
├── DQN/                # DQN baseline (run_DQN.py, dqn.py)
├── A3C/                # A3C baseline (run_A3C.py, a3c.py)
├── DiffRL/             # Proposed diffusion-enabled DRL method
│   ├── run_DiffRL.py
│   └── diffusion/      # Diffusion model and diffusion-SAC agent
└── Greedy/             # Greedy baseline (run_greedy.py, greedy_tasksize.py)
```

## Requirements

- Python 3.14
- PyTorch
- Gym
- NumPy
- SciPy
- NetworkX
- Matplotlib
- Seaborn
- ...

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run from the project root directory:

```bash
# DQN baseline
python methods/DQN/run_DQN.py

# A3C baseline
python methods/A3C/run_A3C.py

# Proposed DiffRL method
python methods/DiffRL/run_DiffRL.py

# Greedy baseline
python methods/Greedy/run_greedy.py
```

## Citation

If you find this code useful, please cite our paper:

```bibtex
@article{yang2026multihop,
  title   = {Multi-Hop Task Offloading Based on Diffusion-Enabled Deep Reinforcement Learning in RSU-Assisted Internet of Vehicles},
  author  = {Yang, Genyuan and Liu, Yihao and Hu, Xiaoying and Li, Wenjuan and Zhao, Weisheng and Wang, Jianbao},
  journal = {(to be added)},
  year    = {2026}
}

```
