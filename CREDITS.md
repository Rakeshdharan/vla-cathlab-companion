# Credits

CathLab Companion builds on these projects. Thank you to their authors. Each is used under its own licence.

| Project | Used for | Link | Licence |
|---|---|---|---|
| robosuite | Simulation framework for the Panda pick task | https://github.com/ARISE-Initiative/robosuite | MIT |
| MuJoCo | Physics engine under robosuite | https://github.com/google-deepmind/mujoco | Apache-2.0 |
| ManiSkill3 | Second Panda engine (PhysX 5), Panda URDF and EE-delta controller | https://github.com/haosulab/ManiSkill | Apache-2.0 |
| SAPIEN | Simulator/renderer under ManiSkill3 | https://github.com/haosulab/SAPIEN | MIT |
| LeRobot | Dataset format (v3), training, policies | https://github.com/huggingface/lerobot | Apache-2.0 |
| SmolVLA | Baseline vision-language-action policy (via LeRobot) | https://huggingface.co/lerobot/smolvla_base | Apache-2.0 |
| ACT | Non-language baseline policy (via LeRobot) | https://github.com/tonyzhaozh/act | MIT |
| NVIDIA Isaac Sim / Isaac Lab | Provided FR3 Duo clinical scene | https://github.com/isaac-sim/IsaacLab | BSD-3-Clause (Isaac Lab); Isaac Sim under NVIDIA licence |
| Hugging Face Hub | Dataset and model hosting | https://huggingface.co | - |
| FastAPI, uvicorn, Starlette | Control plane and worker HTTP APIs | https://github.com/fastapi/fastapi | MIT / BSD-3-Clause |
| pydantic | Configs, contracts, API models | https://github.com/pydantic/pydantic | MIT |
| GitHub Copilot CLI | Default coding agent of the harness | https://github.com/github/copilot-cli | GitHub terms |
| LIBERO | Action/observation conventions (7-D EEF delta, agentview + eye-in-hand) | https://github.com/Lifelong-Robot-Learning/LIBERO | MIT |
| RoboSimStudio | Prompt-to-scene showcase (optional) | https://github.com/robosimstudio/robosimstudio | Apache-2.0 |

Papers and notes:
- Zhou & Zhang, *A tutorial note on collecting simulated data for vision-language-action models*, arXiv:2508.06547
  (no-op filtering, success hold, state recording, validation on aggregation).
- Shukor et al., *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics*, arXiv:2506.01844.
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT), arXiv:2304.13705.
- Liu et al., *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*, arXiv:2306.03310.
- Zhu et al., *robosuite: A Modular Simulation Framework and Benchmark for Robot Learning*, arXiv:2009.12293.
- Tao et al., *ManiSkill3: GPU Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI*, arXiv:2410.00425.

The Siemens Healthineers hackathon materials (brief, provided scene and datasets) are restricted and are not
redistributed here.
