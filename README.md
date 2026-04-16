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
* **Échelle de temps** : Pour ce run, nous avons opté pour une stabilité maximale en choisissant des taux d'apprentissage identiques pour les deux réseaux : `lr_actor = lr_critic = 1e-5`.
* **Horizon de décision** : Nous utilisons `n_layers = 1`, favorisant une boucle de contrôle réactive plutôt qu'une planification complexe en boucle ouverte.

## Évaluation et Efficacité Matérielle

L'ensemble de la recherche d'hyperparamètres a été effectuée sur un Mac Mini M4, permettant d'obtenir des résultats satisfaisants en seulement **15 minutes**. Le run final, nécessaire pour obtenir une IA fine et performante sur 100 000 étapes, dure un peu moins de **2h30**.

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