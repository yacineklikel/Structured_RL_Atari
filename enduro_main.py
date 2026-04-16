version = "260416"

#########################################################
# HYPERPARAMETRES
#########################################################
taille_buffer = int(1e5)
n_steps = int(1e5)
n_envs = 32
batch_size = 256

#hyperparametres de Deep_Mind 2013 (je ne les ai pas touché pour l'instant)
gamma = 0.99
n_skip = 4

#parmis les hyper-parametres de DeepMind 2013, jai seulement modifie
lr_critic = 1e-5  # 2.5e-4 de base masi je voulais le ralentir pour plus de stabilite,
                  # car les courbes de loss etaient tres bruitees sinon pour chaucn 
                  # des 2 reseaux.

#hyperparametres supplementaires pour le Structured RL. Jai trouve ces valeurs en tatonant via le fichier .ipynb joint
lr_actor = 1e-5
temperature_actor = 0.1
sigma_b = 2
n_layers_actions = 1 # nb d'actions que l'actor anticipe a chaque etape (profondeur de l'arbre de decision de l'actor)
                     # (en pratique, il anticipe sur ce nombre d'etapes mais ne realise que la premiere action de son
                     # arbre de decision, puis a la prochaine etape il re-anticipe sur n_layers_actions et ainsi de suite)
                     # en pratique je lai mis a 1 pour converger plus vite mais il est probable quon puisse laugmenter et 
                     # bourinner le n_step pour trouver des comportements encore plus fins.
nb_steps_critic_upgrade = 100 # tous les nb_steps_critic_upgrade pas, on copie les poids de Psi_beta dans Psi_beta_barre 
                              # suivant les notations de l'article, pour stabiliser l'apprentissage du critic
m_samples = 32 # nombre de perturbations de theta que l'on echantillonne a chaque step
entropy_factor = 0.2 # jai ajoute une entropie dans la loss de l'actor pour forcer l'exploration sinon je tombais
                     # sistematiquement dans une politique qui choisissait toujours le meme chemin aveuglement

# sert a pouvoir reduire le champs des actions a 4 actions pour avoir pas trop de combinaisons
ACTION_MAP = {0: 1,  # Accelerate (Tout droit)
              1: 7,  # Accelerate Right
              2: 8,  # Accelerate Left
              3: 0,  # Slowdown
              }
n_actions = len(ACTION_MAP)
n_nodes = 0
for k in range(n_layers_actions):
   n_nodes += n_actions ** (k + 1) # 4 + 16 + 64 + 256 = 340 
                                      # = taille sortie reseau
                                      # = nb de noeuds dans l'arbre de decision de l'actor
n_paths = n_actions ** n_layers_actions # 4^4 = 256



import numpy as np
from tqdm import tqdm 
import random
import os
import copy

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
# Un chemin antiticpe sur n_layers_actions peut etre percu comme une trajectoire de
#  longueur n_layers_actions ou comme 4 noeuds successifs dans un arbre de decision
# On construit ici une matrice de correspondance qui nous permettra de faire le lien entre :
#  - les 256 chemins possibles
#  - les 340 noeuds de l'arbre de decision de l'actor
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

# Jai repris le reseau classique propose en 2015, encore par DeepMind, pour resoudre des jeux Attari
# (on utilise aussi ce rsseau pour l'actor meme si eux n'avaieent qu'un critic)
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
        x = x.view(x.size(0), -1) # Aplatit l'image
        return self.fc(x)

# Buffer bien optimisepour stocker les transitions directement sur le CPU, et y echantiollonner des batches
class GPUTensorBuffer:
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

# Fonction realisant un step de mise a jour pour l'Actor et le Critique en utilisant le batch d'expérience fourni
# C'est la partie la plus importante du code, c'est ici que se trouvent:
#  - la mise a jour du Critique classique (MSE entre Q et y_j)
#  - la mise a jour de l'Actor avec la loss de Fenchel-Young + entropie que j'ai ajoute pour forcer l'exploration
def train_srl_step(
    actor, critic, critic_target,
    opt_actor, opt_critic,
    batch, device, 
    gamma=gamma, sigma_b=sigma_b, temperature_actor=temperature_actor, entropy_factor=entropy_factor, m_samples=m_samples
):
    states, actions, rewards, next_states, dones = batch
    batch_size = states.shape[0]

    # =========================================================================
    # PHASE 1 : Mise à jour du Critique
    # =========================================================================
    with torch.no_grad():
        next_q = critic_target(next_states)
        max_next_q, _ = torch.max(next_q, dim=1)
        target_q = rewards + gamma * (1 - dones) * max_next_q #y_j dans le papier, la cible pour le critic

    critic_all_outputs = critic(states)    
    current_q = critic_all_outputs.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss_critic = F.mse_loss(current_q, target_q)
    
    opt_critic.zero_grad()
    loss_critic.backward()
    opt_critic.step()

    # =========================================================================
    # PHASE 2 : Mise à jour de l'Acteur (Loss de Fenchel-Young / Structured RL)
    # =========================================================================
    theta = actor(states)
    
    noise = torch.randn(batch_size, m_samples, n_nodes, device=device) * sigma_b
    perturbed_theta = theta.unsqueeze(1) + noise 
    path_scores = perturbed_theta[:, :, path_map].sum(dim=-1) # [batch_size, m_samples, 256]
    best_path_ids = torch.argmax(path_scores, dim=-1) # [batch_size, m_samples]
    first_actions = path_map[best_path_ids, 0] 
    with torch.no_grad():
        q_expand = critic_all_outputs.detach().unsqueeze(1).expand(-1, m_samples, -1)
        q_candidates = torch.gather(q_expand, 2, first_actions.unsqueeze(-1)).squeeze(-1)
        q_max, _ = torch.max(q_candidates, dim=1, keepdim=True)
        target_probs = F.softmax((q_candidates - q_max) / temperature_actor, dim=1) 

    # Calcul de la loss de l'Actor selon la formule de Fenchel-Young en 2 termes :
    # TERME 1 : Espérance du max perturbé : E[ max_a (theta_bruite.T @ a) ]
    max_path_scores = torch.max(path_scores, dim=-1)[0] # Shape: [batch_size, m_samples]
    premier_terme = torch.mean(max_path_scores, dim=1).mean() 
    # TERME 2 : - theta.T @ a_critic
    unperturbed_path_scores = theta[:, path_map].sum(dim=-1) # [batch_size, 256]
    scores_for_sampled_paths = torch.gather(unperturbed_path_scores, 1, best_path_ids) 
    deuxieme_terme = -torch.sum(target_probs * scores_for_sampled_paths, dim=1).mean()

    # Calcul de l'entropie de la distribution de probabilités sur les chemins pour forcer l'exploration
    log_path_probs = F.log_softmax(unperturbed_path_scores, dim=-1)
    path_probs = torch.exp(log_path_probs)
    entropy_tensor = -torch.sum(path_probs * log_path_probs, dim=-1).mean()

    # LA LOSS FINALE
    loss_actor = premier_terme + deuxieme_terme - (entropy_factor * entropy_tensor)
    # -------------------------------------------------------------------------

    # Capteurs Passifs pour les graphiques 
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

# Fonction moche qui permet de lancer l'emulateur Atari sans l'ecrire dans mon terminal
# (juste pour avoir un joli terminal avec seulement les tqdm dedans)
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

# Aucune utilite dans le training, c'est juste pour filmer le joueur etvoir comment il se debrouille
def evaluate_and_record(actor, device, step_num, video_folder=f"./videos_{version}"):
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
            action_env = ACTION_MAP[action_nn] #traduction des 4 dim du reseau en les 4 actions permieses parmis les 9
        obs, _ , terminated, truncated, _ = eval_env.step(action_env)
        done = terminated or truncated
    eval_env.close()

if __name__ == '__main__':
    os.makedirs(f"./poids_{version}", exist_ok=True)
    os.makedirs(f"./videos_{version}", exist_ok=True)
    os.makedirs(f"./plots_{version}", exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Utilisation de l'appareil : {device}")

    actor = NatureCNN(n_nodes).to(device)
    critic = NatureCNN(n_actions).to(device)         # Les poids Psi_beta du papier
    critic_target = copy.deepcopy(critic).to(device) # Les poids Pss_beta_barre du papier
    
    opt_actor = optim.Adam(actor.parameters(), lr=lr_actor)
    opt_critic = optim.Adam(critic.parameters(), lr=lr_critic)

    path_map = PATH_INDICES.to(device)

    print(f"Démarrage des {n_envs} environnements parallèles...")
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(n_envs)])
    replay_buffer = GPUTensorBuffer(capacity=taille_buffer, n_envs=n_envs, device=device)
    obs, infos = envs.reset()
    history_loss_c = np.zeros(n_steps) # pour le .ipynb
    history_loss_a = np.zeros(n_steps)
    history_entropy = np.zeros(n_steps)
    history_confidence = np.zeros(n_steps)

    print("Début de l'entraînement !")
    # =========================================================================
    # PHASE DE WARM-UP : Remplissage du buffer avec des actions aléatoires
    # =========================================================================
    current_stack = torch.zeros((n_envs, n_skip, 84, 84), dtype=torch.uint8, device=device)
    obs, infos = envs.reset()
    obs_gpu = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
    current_stack = obs_gpu.unsqueeze(1).repeat(1, 4, 1, 1) 
    warmup_ratio = 1.0 # pour preremplir 100% du buffer
    warmup_steps = int((taille_buffer // n_envs) * warmup_ratio)
    print(f"\nPhase d'échauffement : Pré-remplissage ({warmup_steps * n_envs} transitions)...")
    # Tableaux pour suivre l'état (l'action et le verrou) de chaque environnement (32 au total)
    current_actions = [0] * n_envs
    action_locks = [0] * n_envs

    for _ in tqdm(range(warmup_steps)):
        # Choix d'actions purement aléatoires pendant l'échauffement
        action_nn = [random.randint(0, n_actions - 1) for _ in range(n_envs)]
        action_env = [ACTION_MAP[a] for a in action_nn]
        action_env = [1 for _ in range(n_envs)] #juste pour le warmup, on met que l'action "accelerate" pour que le buffer soit rempli de transitions un peu plus interessantes que si on mettait des actions completement aleatoires

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


    print("Échauffement terminé. Le Buffer est amorcé avec succès !")
    # =========================================================================

    print("\nDébut de l'entraînement !")
    next_save_step = 400

    for step in tqdm(range(n_steps)):
        with torch.no_grad():
            theta = actor(current_stack)
            path_scores = theta[:, path_map].sum(dim=-1) # [n_envs, 256]
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
        
        obs = next_obs
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)
            
            loss_c, loss_a, entropy, max_confidence = train_srl_step(
                actor, critic, 
                critic_target,
                opt_actor, opt_critic,
                batch, device,
                sigma_b=sigma_b
            )
            
            history_loss_c[step] = loss_c
            history_loss_a[step] = loss_a
            history_entropy[step] = entropy
            history_confidence[step] = max_confidence

            if step % nb_steps_critic_upgrade == 0:
                critic_target.load_state_dict(critic.state_dict())                
        # 6. FILMER LA PROGRESSION
        if step == 0 or (step + 1) >= next_save_step:
            print(f"\nEnregistrement de la vidéo à l'étape {step + 1}...")
            evaluate_and_record(actor, device, step + 1)
            torch.save(actor.state_dict(), f"./poids_{version}/actor_{step + 1}.pth")
            torch.save(critic.state_dict(), f"./poids_{version}/critic_{step + 1}.pth")
            if step > 0:
                next_save_step *= 2
                
        del next_obs_gpu_uint8
        del next_obs_float
        if 'batch' in locals():
            del batch
        
        if step % 100 == 0:
            import gc
            gc.collect() 
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    envs.close()
    print("\nEntraînement terminé ! Sauvegarde du cerveau de l'IA en cours...")
    torch.save(actor.state_dict(), f"./poids_{version}/actor_{step + 1}.pth")
    torch.save(critic.state_dict(), f"./poids_{version}/critic_{step + 1}.pth")
    np.save(f"./poids_{version}/history_loss_c.npy", history_loss_c)
    np.save(f"./poids_{version}/history_loss_a.npy", history_loss_a)
    np.save(f"./poids_{version}/history_entropy.npy", history_entropy)
    np.save(f"./poids_{version}/history_confidence.npy", history_confidence)
    evaluate_and_record(actor, device, "final")
    print("Modèle sauvegardé sous 'mon_modele_enduro.pth' !")
