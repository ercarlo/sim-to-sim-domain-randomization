import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from env.custom_hopper import *


def extract_latents(model, obs):   #returns  Latent representation produced by the policy network.  latent : np.ndarray
  
    obs_tensor = torch.as_tensor(obs).float().unsqueeze(0)
    with torch.no_grad():  # Desactiva el cálculo de gradientes (no estamos entrenando)
        latent_pi, _ = model.policy.mlp_extractor(obs_tensor)
    return latent_pi.squeeze(0).cpu().numpy()


def collect_representations(model_path, env_id, n_episodes=20, max_steps=500):
    env = gym.make(env_id)
    model = PPO.load(model_path)

    latents = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        steps = 0
        done = False

        while not done and steps < max_steps:
            latent = extract_latents(model, obs)
            latents.append(latent)

            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

    env.close()
    return np.array(latents)


def project_and_plot(source_latents,target_latents,title,ax_pca,ax_tsne):
    """
    Plot PCA and t-SNE projections into given matplotlib axes.
    """

    # Combine data
    X = np.vstack([source_latents, target_latents])
    y = np.array([0]*len(source_latents) + [1]*len(target_latents))

    # ---------- PCA ----------
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    ax_pca.scatter(
        X_pca[y == 0, 0],
        X_pca[y == 0, 1],
        alpha=0.5,
        label="Source"
    )
    ax_pca.scatter(
        X_pca[y == 1, 0],
        X_pca[y == 1, 1],
        alpha=0.5,
        label="Target"
    )
    ax_pca.set_title(title)
    ax_pca.set_xlabel("PC1")
    ax_pca.set_ylabel("PC2")

    # ---------- t-SNE ----------
    tsne = TSNE(n_components=2, perplexity=30, init="pca")
    X_tsne = tsne.fit_transform(X)

    ax_tsne.scatter(
        X_tsne[y == 0, 0],
        X_tsne[y == 0, 1],
        alpha=0.5,
        label="Source"
    )
    ax_tsne.scatter(
        X_tsne[y == 1, 0],
        X_tsne[y == 1, 1],
        alpha=0.5,
        label="Target"
    )
    ax_tsne.set_xlabel("Dim 1")
    ax_tsne.set_ylabel("Dim 2")

if __name__ == "__main__":

    MODELS = [
        ("Baseline (no DR)", "NoDRModels/ppo_train-source_seed-0"),
        ("DR all ±20%", "GoodModels/ppo_train-source_seed-0"),
        ("DR all ±40%", "BadDRModels/ppo_train-source_seed-0"),
    ]

    # Create figure: 2 rows (PCA, t-SNE) x 3 columns (models)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(18, 10),
        sharex=False,
        sharey=False
    )

    for col, (name, path) in enumerate(MODELS):
        print(f"Processing {name}")

        src_latents = collect_representations(
            model_path=path,
            env_id="CustomHopper-source-v0"
        )

        tgt_latents = collect_representations(
            model_path=path,
            env_id="CustomHopper-target-v0"
        )

        ax_pca = axes[0, col]
        ax_tsne = axes[1, col]

        project_and_plot(
            src_latents,
            tgt_latents,
            title=name,
            ax_pca=ax_pca,
            ax_tsne=ax_tsne
        )

    # Legend only once (clean)
    axes[0, 0].legend(loc="best")

    plt.suptitle("Latent Representation Analysis (Source vs Target)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()