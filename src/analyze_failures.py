import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from collections import defaultdict

from env.custom_hopper import *  # registra los entornos


# ==========================
# Evaluation configuration
# ==========================
MAX_EPISODE_STEPS = 3000
NEAR_FALL_Z = 0.75  # threshold for "almost falling"


# ==========================
# Stability analysis
# ==========================
def analyze_model(model_path, n_episodes=50, seed=0):
    env = gym.make(
        "CustomHopper-target-v0",
        max_episode_steps=MAX_EPISODE_STEPS
    )
    env.reset(seed=seed)

    # IMPORTANT: no DR during evaluation
    env.unwrapped.enable_dr = False

    model = PPO.load(model_path)

    metrics = defaultdict(list)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        zs = []
        angles = []
        xvels = []

        steps = 0
        terminated_flag = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            zs.append(float(obs[0]))        # torso height
            angles.append(float(obs[1]))    # torso angle
            xvels.append(float(info["x_velocity"]))

            steps += 1
            if terminated:
                terminated_flag = True

        # ---- Per-episode metrics ----
        metrics["episode_length"].append(steps)
        metrics["min_z"].append(np.min(zs))
        metrics["angle_std"].append(np.std(angles))
        metrics["mean_x_velocity"].append(np.mean(xvels))
        metrics["near_falls"].append(sum(z < NEAR_FALL_Z for z in zs))

        if terminated_flag:
            metrics["time_to_fail"].append(steps)
            metrics["timeout"].append(0)
        else:
            metrics["time_to_fail"].append(MAX_EPISODE_STEPS)
            metrics["timeout"].append(1)

    env.close()
    return metrics


# ==========================
# Reporting
# ==========================
def print_summary(name, metrics):
    print(f"\n===== {name} =====")
    print(f"Timeout rate        : {100*np.mean(metrics['timeout']):.1f}%")
    print(f"Avg episode length  : {np.mean(metrics['episode_length']):.1f}")
    print(f"Avg time-to-fail    : {np.mean(metrics['time_to_fail']):.1f}")
    print(f"Avg min torso z     : {np.mean(metrics['min_z']):.3f}")
    print(f"Avg angle std       : {np.mean(metrics['angle_std']):.3f}")
    print(f"Avg near-falls      : {np.mean(metrics['near_falls']):.1f}")
    print(f"Avg forward velocity: {np.mean(metrics['mean_x_velocity']):.2f}")


# ==========================
# Main
# ==========================
if __name__ == "__main__":

    MODELS = {
        "Baseline (no DR)": "NoDRModels/ppo_train-source_seed-0",
        "DR all ±20% (best)": "GoodModels/ppo_train-source_seed-0",  # rename if needed
        "DR all ±40%": "BadDRModels/ppo_train-source_seed-0",          # optional
    }

    for name, path in MODELS.items():
        metrics = analyze_model(
            model_path=path,
            n_episodes=50,
            seed=123
        )
        print_summary(name, metrics)