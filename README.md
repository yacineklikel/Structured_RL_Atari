# Structured Reinforcement Learning on Atari Enduro

This repository is an experimental study that applies a recent method from the literature, [Structured Reinforcement Learning (SRL)](references/2505.19053.pdf), to a classic and demanding benchmark: the Atari environment popularised by DeepMind's foundational [DQN](references/1312.5602.pdf) work.

The goal is to assess whether a combinatorial optimisation layer, trained end-to-end inside an Actor-Critic architecture, can cope with the real-time dynamics of a racing game.

## Method

Both the actor and the critic use a **NatureCNN**-style convolutional network to extract visual features. The discrete action layer is made differentiable by minimising a **Fenchel-Young loss** with stochastic perturbations:

$$\mathcal{L}_{FY}(\theta; \hat{a}) = \mathbb{E}_{Z \sim \mathcal{N}(0, \sigma_b)} \left[ \max_{a \in \mathcal{A}} (\theta + Z)^\top a \right] - \theta^\top \hat{a}$$

The final actor loss adds an entropy regularisation term to keep exploring:

$$\mathcal{L}_{Actor} = \frac{1}{M}\sum_{i=1}^{M} \max_{a} (\theta + Z_i)^\top a - \theta^\top \hat{a} - \lambda \mathcal{H}(\pi_\theta)$$

The target $\hat{a}$ is obtained by applying a softmax to the critic's Q-values.

## Architecture and hyperparameter choices

* **Inherited from DQN**: the discount factor `gamma = 0.99` and `frame_skip = 4` are kept from DeepMind's original work.
* **Learning rates**: for the first tests and the intermediate run, I prioritised stability with identical, fixed learning rates (`lr_actor = lr_critic = 1e-5`). The final run adds a schedule on the actor's learning rate.
* **Planning horizon**: `n_layers_actions = 1`, which favours a reactive control loop over complex open-loop planning.
* **Action space**: reduced to 4 actions (accelerate, accelerate left, accelerate right, slow down).

## Training and results

* **Initial tests (~15 min)**: after tuning the first hyperparameters, the agent already showed avoidance behaviour early in training.
* **Intermediate run ([train_100k_steps.py](./train_100k_steps.py))**: 100,000 steps (~2.5 h) bring the agent close to average human level. It drives well on the green ground but struggles when the colours change (night / snow).
  * See the [video after 10^5 steps](./Video_100k.mp4).
* **Full run ([train_1M_steps.py](./train_1M_steps.py))**: 1,000,000 steps (~27 h). This run linearly schedules `sigma_b`, `entropy_factor` and `lr_actor` to refine the policy towards the end of training.
  * **Performance**: the agent now handles all terrain types and passes about **750 cars** per episode, above the 470 reported for DQN in [DeepMind's 2013 paper](./references/1312.5602.pdf).
  * **Note**: training was stopped after 27 h because that was enough to validate the algorithm. Higher scores are probably reachable with longer training.
  * See the [video after 10^6 steps](./Video_1M.mp4) for the final result.

## Repository structure

* `train_100k_steps.py`: intermediate training run (fixed hyperparameters).
* `train_1M_steps.py`: full training run (scheduled hyperparameters).
* `Video_100k.mp4`, `Video_1M.mp4`: gameplay videos of the agent after 10^5 and 10^6 steps.
* `weights/`: sample actor / critic checkpoints (steps 1, 400 and 800).
* `references/`: source papers ([Mnih et al., 2013](references/1312.5602.pdf) and [Hoppe et al., 2025](references/2505.19053.pdf)).

Each training script writes its outputs to folders tagged with the run name (not tracked by git):

* `weights_<run>/`: actor and critic checkpoints, saved at step 1, 400, 800, 1600, … and at the end of training.
* `videos_<run>/`: MP4 recordings of the agent at each checkpoint.
* `metrics_<run>/`: NumPy arrays of the critic loss, actor loss, policy entropy and critic confidence at each step. I used them to tune the hyperparameters.

## Installation and usage

1. **Dependencies** (tested on an Apple Silicon Mac with Python 3.12; uses the MPS backend if available, otherwise the CPU):
   ```bash
   pip install -r requirements.txt
   ```

2. **Training**:
   ```bash
   python train_100k_steps.py   # ~2.5 h
   python train_1M_steps.py     # ~27 h
   ```
