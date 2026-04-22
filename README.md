# Exploration du Structured Reinforcement Learning sur Atari Enduro

Ce dépôt présente un travail expérimental visant à confronter une méthode récente de la littérature — le [Structured Reinforcement Learning (SRL)](references/2505.19053.pdf) — à un banc d'essai classique et exigeant : l'environnement Atari, popularisé par les travaux fondateurs de [DeepMind (DQN)](references/1312.5602.pdf).

Notre objectif est d'étudier la viabilité d'une couche d'optimisation combinatoire intégrée de bout en bout au sein d'une architecture Actor-Critic, face aux dynamiques temps réel d'un jeu de course.

## Méthodologie et Équations

Nous utilisons l'architecture de base **NatureCNN** pour l'extraction de caractéristiques visuelles. La différenciation de la couche d'actions discrètes repose sur la minimisation d'une **perte de Fenchel-Young** couplée à des perturbations stochastiques :

$$\mathcal{L}_{FY}(\theta; \hat{a}) = \mathbb{E}_{Z \sim \mathcal{N}(0, \sigma_b)} \left[ \max_{a \in \mathcal{A}} (\theta + Z)^\top a \right] - \theta^\top \hat{a}$$

La perte finale de l'Acteur inclut une régularisation entropique pour maintenir l'exploration :

$$\mathcal{L}_{Acteur} = \frac{1}{M}\sum_{i=1}^{M} \max_{a} (\theta + Z_i)^\top a - \theta^\top \hat{a} - \lambda \mathcal{H}(\pi_\theta)$$

Le signal de guidage $\hat{a}$ est généré par un Softmax sur les Q-Values du Critique.

## Choix d'Architecture et d'Hyperparamètres

* **Héritage** : Nous conservons le facteur de discount `gamma = 0.99` et le `frame_skip = 4` des travaux originaux de DeepMind.
* **Échelle de temps** : Pour les premiers tests et l'entraînement intermédiaire, nous avons opté pour une stabilité maximale avec des taux d'apprentissage identiques et fixes (`lr_actor = lr_critic = 1e-5`), avant d'introduire un *scheduling* dynamique pour l'Acteur dans le run final.
* **Horizon de décision** : Nous utilisons `n_layers = 1`, favorisant une boucle de contrôle réactive plutôt qu'une planification complexe en boucle ouverte.

## Résultats et Entraînement

L'évolution de l'agent a été documentée à travers différentes phases d'expérimentation :

* **Phase de test initiale (~15 min) :** L'ajustement des premiers hyperparamètres a permis d'observer des comportements d'évitement dès le début de l'apprentissage.
* **Entraînement intermédiaire ([train_100k_steps.py](./train_100k_steps.py)) :** Ce run de 100 000 étapes (environ 2h30) permet d'atteindre un niveau proche de celui d'un humain moyen. L'agent est performant sur sol vert, bien qu'il éprouve des difficultés lors des inversions de couleurs (nuit/neige).
    * *Voir la [vidéo à 10^5 étapes](./Video_1e5.mp4) pour observer ce comportement.*
* **Entraînement complet ([train_1M_steps.py](./train_1M_steps.py)) :** Un run de 1 000 000 d'étapes (environ 27h) a été effectué. Pour cette phase, une planification (*scheduling*) des paramètres `sigma_b`, `entropy_factor` et `lr_actor` a été mise en place pour affiner la politique en fin d'apprentissage.
    * **Performances :** L'agent reconnaît désormais les différents terrains et atteint des scores d'environ **750 voitures doublées**, dépassant les 470 rapportés par le modèle DQN dans l'article de [DeepMind (2013)](./references/1312.5602.pdf).
    * **Note :** Le training a été stoppé à 27h car ce délai suffisait à valider le fonctionnement de l'algorithme, bien que des scores plus élevés soient probablement atteignables en prolongeant le calcul.
    * *Voir la [vidéo à 10^6 étapes](./Video_1e6.mp4) pour le résultat final.*

## Organisation du Dépôt

* **references/** : Contient les articles sources ([Mnih et al., 2013](references/1312.5602.pdf) et [Hoppe et al., 2025](references/2505.19053.pdf)).
* **poids_/** : Sauvegardes des modèles (Actor et Critic).
* **videos_/** : Enregistrements MP4 de la progression de l'agent.
* **metrics_/** : Historiques des tableaux Numpy (Loss, Entropie, Confiance) qui ont permis de trouver les bons paramètres grâce aux plots réalisés dans le notebook `.ipynb`.
* **plots_/** : Visualisations de synthèse.

## Installation et Utilisation

1. **Dépendances** :
   ```bash
   pip install -r requirements.txt