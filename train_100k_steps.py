run_name = "100k"  # tag used for the output folders (weights_100k/, videos_100k/, metrics_100k/)

#########################################################
# HYPERPARAMETERS
#########################################################
buffer_size = int(1e5)
n_steps = int(1e5)
n_envs = 32
batch_size = 256

# Hyperparameters from DeepMind 2013 (left unchanged)
gamma = 0.99
n_skip = 4

# The only DeepMind 2013 hyperparameter I changed:
lr_critic = 1e-5  # 2.5e-4 originally, lowered for stability: otherwise the loss
                  # curves of both networks were very noisy.

# Additional Structured RL hyperparameters, tuned by trial and error
# by looking at the loss / entropy / confidence curves.
lr_actor = 1e-5
temperature_actor = 0.1
sigma_b = 2
n_layers_actions = 1  # number of actions the actor plans ahead at each step (depth of its decision tree).
                      # The actor plans over this horizon but only executes the first action of the
                      # best path, then re-plans at the next step (receding horizon).
                      # Set to 1 for faster convergence; a deeper tree combined with more training
                      # steps could probably produce finer behaviours.
nb_steps_critic_upgrade = 100  # every nb_steps_critic_upgrade steps, the weights psi_beta are copied into
                               # psi_beta_bar (paper notation) to stabilise critic training
m_samples = 32  # number of perturbations of theta sampled at each step
entropy_factor = 0.2  # entropy bonus added to the actor loss to force exploration; without it the policy
                      # systematically collapsed onto always choosing the same path

# Restricts the action space to 4 actions to keep the number of combinations small
ACTION_MAP = {0: 1,  # Accelerate (straight)
              1: 7,  # Accelerate Right
              2: 8,  # Accelerate Left
              3: 0,  # Slowdown
              }
n_actions = len(ACTION_MAP)
n_nodes = 0
for k in range(n_layers_actions):
    n_nodes += n_actions ** (k + 1)  # n_actions + n_actions^2 + ... (e.g. 4 + 16 + 64 + 256 = 340 for a depth of 4)
                                     # = actor output size
                                     # = number of nodes in the actor's decision tree
n_paths = n_actions ** n_layers_actions  # number of leaves / paths (e.g. 4^4 = 256 for a depth of 4)



import numpy as np
from tqdm import tqdm
import os
import copy
import gc
import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from gymnasium.wrappers import AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation
import ale_py
gym.register_envs(ale_py)


# =========================================================================
# A path planned over n_layers_actions steps can be seen either as a trajectory
# of n_layers_actions actions, or as n_layers_actions successive nodes in a decision tree.
# Here we build a lookup table linking:
#  - the n_paths possible paths
#  - the n_nodes nodes of the actor's decision tree
# =========================================================================
paths = []
first_leaf = sum(n_actions**k for k in range(1, n_layers_actions))
num_leaves = n_actions ** n_layers_actions
for leaf in range(first_leaf, first_leaf + num_leaves):
    path = [leaf]
    current = leaf
    for _ in range(n_layers_actions - 1):
        current = (current - n_actions) // n_actions
        path.append(current)
    paths.append(path[::-1])
PATH_INDICES = torch.tensor(paths, dtype=torch.long)

# NatureCNN-style network, inspired by DeepMind's 2015 Atari agent (Mnih et al., Nature 2015).
# It is also used for the actor here, even though DQN only had a critic.
class NatureCNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, 512), nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # flatten the feature maps
        return self.fc(x)

# Memory-efficient replay buffer: stores single uint8 frames on the CPU (instead of full
# 4-frame stacks) and rebuilds the stacked states on the target device at sampling time.
class ReplayBuffer:
    def __init__(self, capacity, n_envs, device):
        self.n_envs = n_envs
        self.max_steps = capacity // n_envs
        self.ptr = 0
        self.size = 0
        self.device = device

        self.frames = torch.zeros((self.max_steps, n_envs, 84, 84), dtype=torch.uint8)
        self.actions = torch.zeros((self.max_steps, n_envs), dtype=torch.int64)
        self.rewards = torch.zeros((self.max_steps, n_envs), dtype=torch.float32)
        self.dones = torch.zeros((self.max_steps, n_envs), dtype=torch.float32)

    def __len__(self):
        return self.size * self.n_envs

    def push(self, obs_uint8, actions, rewards, dones):
        if isinstance(obs_uint8, torch.Tensor):
            self.frames[self.ptr] = obs_uint8.cpu()
        else:
            self.frames[self.ptr] = torch.from_numpy(obs_uint8)

        self.actions[self.ptr] = torch.tensor(actions, dtype=torch.int64)
        self.rewards[self.ptr] = torch.from_numpy(rewards).to(dtype=torch.float32)
        self.dones[self.ptr] = torch.from_numpy(dones).to(dtype=torch.float32)

        self.ptr = (self.ptr + 1) % self.max_steps
        self.size = min(self.size + 1, self.max_steps)

    def sample(self, batch_size):
        t_indices = torch.randint(3, self.size - 1, (batch_size,))
        env_indices = torch.randint(0, self.n_envs, (batch_size,))

        if self.size == self.max_steps:
            distance_to_ptr = (self.ptr - t_indices) % self.max_steps
            t_indices[distance_to_ptr <= 4] -= 5

        s0 = self.frames[t_indices - 3, env_indices]
        s1 = self.frames[t_indices - 2, env_indices]
        s2 = self.frames[t_indices - 1, env_indices]
        s3 = self.frames[t_indices, env_indices]
        ns3 = self.frames[t_indices + 1, env_indices]

        states = torch.stack([s0, s1, s2, s3], dim=1).to(torch.float32).to(self.device) / 255.0
        next_states = torch.stack([s1, s2, s3, ns3], dim=1).to(torch.float32).to(self.device) / 255.0

        actions = self.actions[t_indices, env_indices].to(self.device)
        rewards = self.rewards[t_indices, env_indices].to(self.device)
        dones = self.dones[t_indices, env_indices].to(self.device)

        return states, actions, rewards, next_states, dones

# One update step for both the actor and the critic on the given batch of experience.
# This is the core of the algorithm:
#  - standard critic update (MSE between Q and the target y_j)
#  - actor update with the Fenchel-Young loss, plus an entropy bonus to force exploration
def train_srl_step(
    actor, critic, critic_target,
    opt_actor, opt_critic,
    batch, path_map, device,
    gamma=gamma, sigma_b=sigma_b, temperature_actor=temperature_actor, entropy_factor=entropy_factor, m_samples=m_samples
):
    states, actions, rewards, next_states, dones = batch
    batch_size = states.shape[0]

    # =========================================================================
    # PHASE 1: Critic update
    # =========================================================================
    with torch.no_grad():
        next_q = critic_target(next_states)
        max_next_q, _ = torch.max(next_q, dim=1)
        target_q = rewards + gamma * (1 - dones) * max_next_q  # y_j in the paper: the critic's target

    critic_all_outputs = critic(states)
    current_q = critic_all_outputs.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss_critic = F.mse_loss(current_q, target_q)

    opt_critic.zero_grad()
    loss_critic.backward()
    opt_critic.step()

    # =========================================================================
    # PHASE 2: Actor update (Fenchel-Young loss / Structured RL)
    # =========================================================================
    theta = actor(states)

    noise = torch.randn(batch_size, m_samples, n_nodes, device=device) * sigma_b
    perturbed_theta = theta.unsqueeze(1) + noise
    path_scores = perturbed_theta[:, :, path_map].sum(dim=-1)  # [batch_size, m_samples, n_paths]
    best_path_ids = torch.argmax(path_scores, dim=-1)  # [batch_size, m_samples]
    first_actions = path_map[best_path_ids, 0]
    with torch.no_grad():
        q_expand = critic_all_outputs.detach().unsqueeze(1).expand(-1, m_samples, -1)
        q_candidates = torch.gather(q_expand, 2, first_actions.unsqueeze(-1)).squeeze(-1)
        q_max, _ = torch.max(q_candidates, dim=1, keepdim=True)
        target_probs = F.softmax((q_candidates - q_max) / temperature_actor, dim=1)

    # Actor loss, following the Fenchel-Young formula in two terms:
    # TERM 1: expectation of the perturbed max: E[ max_a (theta_perturbed^T a) ]
    max_path_scores = torch.max(path_scores, dim=-1)[0]  # [batch_size, m_samples]
    first_term = torch.mean(max_path_scores, dim=1).mean()
    # TERM 2: - theta^T a_critic
    unperturbed_path_scores = theta[:, path_map].sum(dim=-1)  # [batch_size, n_paths]
    scores_for_sampled_paths = torch.gather(unperturbed_path_scores, 1, best_path_ids)
    second_term = -torch.sum(target_probs * scores_for_sampled_paths, dim=1).mean()

    # Entropy of the probability distribution over paths, to force exploration
    log_path_probs = F.log_softmax(unperturbed_path_scores, dim=-1)
    path_probs = torch.exp(log_path_probs)
    entropy_tensor = -torch.sum(path_probs * log_path_probs, dim=-1).mean()

    # FINAL LOSS
    loss_actor = first_term + second_term - (entropy_factor * entropy_tensor)
    # -------------------------------------------------------------------------

    # Passive monitoring metrics for the plots (no gradient)
    with torch.no_grad():
        critic_probs = F.softmax(critic_all_outputs / temperature_actor, dim=1)
        max_confidence = torch.max(critic_probs, dim=1)[0].mean().item()
        log_path_probs_passive = F.log_softmax(unperturbed_path_scores, dim=-1)
        path_probs_passive = torch.exp(log_path_probs_passive)
        entropy = -torch.sum(path_probs_passive * log_path_probs_passive, dim=-1).mean().item()

    opt_actor.zero_grad()
    loss_actor.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 2.0)
    opt_actor.step()

    return loss_critic.item(), loss_actor.item(), entropy, max_confidence

# Creates an Atari environment while silencing the emulator's console output
# (keeps the terminal clean, with only the tqdm progress bars)
def make_env():
    def _init():
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        try:
            env = gym.make("ALE/Enduro-v5", frameskip=1)
            env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=n_skip)
        finally:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(devnull)
            os.close(old_stdout)
            os.close(old_stderr)
        return env
    return _init

# Not used for training: plays one greedy episode and records it as a video
# to check how the agent behaves
def evaluate_and_record(actor, path_map, device, step_num, video_folder=f"./videos_{run_name}"):
    eval_env = gym.make("ALE/Enduro-v5", render_mode="rgb_array", frameskip=1)
    eval_env = AtariPreprocessing(eval_env, screen_size=84, grayscale_obs=True, frame_skip=n_skip)
    eval_env = FrameStackObservation(eval_env, stack_size=n_skip)

    eval_env = RecordVideo(eval_env, video_folder, episode_trigger=lambda x: True, name_prefix=f"step_{step_num}")
    obs, _ = eval_env.reset()
    done = False

    while not done:
        obs_array = np.array(obs.__array__())
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
        with torch.no_grad():
            theta = actor(obs_tensor)
            path_scores = theta[:, path_map].sum(dim=-1)
            best_path_ids = torch.argmax(path_scores, dim=1)
            action_nn = path_map[best_path_ids, 0].item()
            action_env = ACTION_MAP[action_nn]
        obs, _ , terminated, truncated, _ = eval_env.step(action_env)
        done = terminated or truncated

    eval_env.close()

    # =========================================================================
    # CLEANUP: rename the recorded files to keep the video folder tidy
    # =========================================================================
    for file_path in glob.glob(os.path.join(video_folder, f"step_{step_num}*episode*")):
        # Rename the video to a clean name
        if file_path.endswith(".mp4"):
            clean_name = os.path.join(video_folder, f"step_{step_num}.mp4")
            # Overwrite the video if the clean name already exists
            if os.path.exists(clean_name):
                os.remove(clean_name)
            os.rename(file_path, clean_name)
        # Discard the metadata file
        elif file_path.endswith(".meta.json"):
            os.remove(file_path)
    # =========================================================================



if __name__ == '__main__':
    os.makedirs(f"./weights_{run_name}", exist_ok=True)
    os.makedirs(f"./videos_{run_name}", exist_ok=True)
    os.makedirs(f"./metrics_{run_name}", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    actor = NatureCNN(n_nodes).to(device)
    critic = NatureCNN(n_actions).to(device)          # weights psi_beta in the paper
    critic_target = copy.deepcopy(critic).to(device)  # weights psi_beta_bar in the paper

    opt_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    path_map = PATH_INDICES.to(device)

    print(f"Starting {n_envs} parallel environments...")
    envs = gym.vector.SyncVectorEnv([make_env() for _ in range(n_envs)])
    replay_buffer = ReplayBuffer(capacity=buffer_size, n_envs=n_envs, device=device)
    history_loss_c = np.zeros(n_steps)  # metrics saved for offline analysis
    history_loss_a = np.zeros(n_steps)
    history_entropy = np.zeros(n_steps)
    history_confidence = np.zeros(n_steps)

    # =========================================================================
    # WARM-UP: pre-fill the replay buffer before training.
    # Only the "Accelerate" action is played, so that the buffer contains more
    # useful transitions than with fully random actions.
    # =========================================================================
    obs, infos = envs.reset()
    obs_gpu = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
    current_stack = obs_gpu.unsqueeze(1).repeat(1, 4, 1, 1)
    warmup_ratio = 1.0  # fraction of the buffer to pre-fill
    warmup_steps = int((buffer_size // n_envs) * warmup_ratio)
    print(f"\nWarm-up: pre-filling the buffer ({warmup_steps * n_envs} transitions)...")
    warmup_action = 0  # index of "Accelerate" in ACTION_MAP

    for _ in tqdm(range(warmup_steps)):
        action_nn = [warmup_action] * n_envs  # the stored action matches the action actually played
        action_env = [ACTION_MAP[a] for a in action_nn]

        next_obs, rewards, terminateds, truncateds, infos = envs.step(action_env)
        dones = np.logical_or(terminateds, truncateds)

        next_obs_gpu_uint8 = torch.from_numpy(next_obs).to(dtype=torch.uint8, device=device)
        replay_buffer.push(next_obs_gpu_uint8, action_nn, rewards, dones)

        current_stack = current_stack.roll(shifts=-1, dims=1)
        next_obs_float = next_obs_gpu_uint8.to(dtype=torch.float32, device=device) / 255.0
        current_stack[:, -1, :, :] = next_obs_float

        dones_tensor = torch.from_numpy(dones).to(dtype=torch.bool, device=device)
        if dones_tensor.any():
            current_stack[dones_tensor] = next_obs_float[dones_tensor].unsqueeze(1).repeat(1, 4, 1, 1)


    print("Warm-up done: the replay buffer is filled.")
    # =========================================================================

    print("\nStarting training!")
    next_save_step = 400

    for step in tqdm(range(n_steps)):
        with torch.no_grad():
            theta = actor(current_stack)
            path_scores = theta[:, path_map].sum(dim=-1)  # [n_envs, n_paths]
            action_probs = F.softmax(path_scores, dim=1)
            best_path_ids = torch.multinomial(action_probs, num_samples=1).squeeze(1)

            action_nn = path_map[best_path_ids, 0].cpu().numpy().tolist()

        action_env = [ACTION_MAP[a] for a in action_nn]
        next_obs, rewards, terminateds, truncateds, infos = envs.step(action_env)
        dones = np.logical_or(terminateds, truncateds)

        next_obs_gpu_uint8 = torch.from_numpy(next_obs).to(dtype=torch.uint8, device=device)
        replay_buffer.push(next_obs_gpu_uint8, action_nn, rewards, dones)
        current_stack = current_stack.roll(shifts=-1, dims=1)
        next_obs_float = next_obs_gpu_uint8.to(dtype=torch.float32, device=device) / 255.0
        current_stack[:, -1, :, :] = next_obs_float
        dones_tensor = torch.from_numpy(dones).to(dtype=torch.bool, device=device)
        if dones_tensor.any():
            current_stack[dones_tensor] = next_obs_float[dones_tensor].unsqueeze(1).repeat(1, 4, 1, 1)

        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)

            loss_c, loss_a, entropy, max_confidence = train_srl_step(
                actor, critic,
                critic_target,
                opt_actor, opt_critic,
                batch, path_map, device,
                sigma_b=sigma_b
            )

            history_loss_c[step] = loss_c
            history_loss_a[step] = loss_a
            history_entropy[step] = entropy
            history_confidence[step] = max_confidence

            if step % nb_steps_critic_upgrade == 0:
                critic_target.load_state_dict(critic.state_dict())
        # Record a video and save checkpoints (at step 1, then 400, 800, 1600, ...)
        if step == 0 or (step + 1) >= next_save_step:
            print(f"\nRecording video at step {step + 1}...")
            evaluate_and_record(actor, path_map, device, step + 1)
            torch.save(actor.state_dict(), f"./weights_{run_name}/actor_{step + 1}.pth")
            torch.save(critic.state_dict(), f"./weights_{run_name}/critic_{step + 1}.pth")
            if step > 0:
                next_save_step *= 2

        del next_obs_gpu_uint8
        del next_obs_float
        if 'batch' in locals():
            del batch

        # Periodically free memory (MPS cache)
        if step % 100 == 0:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    envs.close()
    print("\nTraining finished! Saving the model...")
    torch.save(actor.state_dict(), f"./weights_{run_name}/actor_{n_steps}.pth")
    torch.save(critic.state_dict(), f"./weights_{run_name}/critic_{n_steps}.pth")

    # Save the metrics for offline analysis
    np.save(f"./metrics_{run_name}/loss_c.npy", history_loss_c)
    np.save(f"./metrics_{run_name}/loss_a.npy", history_loss_a)
    np.save(f"./metrics_{run_name}/entropy.npy", history_entropy)
    np.save(f"./metrics_{run_name}/confidence.npy", history_confidence)

    evaluate_and_record(actor, path_map, device, n_steps)
    print(f"Model saved in ./weights_{run_name}/")
