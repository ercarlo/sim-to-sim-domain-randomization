"""
Training script for Hopper (PPO) with optional Domain Randomization (UDR-like).

Notes:
- We do NOT pass domain_randomization/dr_percent as kwargs to gym.make(),
  because CustomHopper __init__ in the template doesn't define those kwargs.
- Instead, we configure DR through env.unwrapped.* attributes.
"""

import os
import argparse
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from env.custom_hopper import *  # registers the envs


def make_env(env_id: str, seed: int):
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def configure_dr(env, dr: bool, dr_percent: float, dr_thigh: bool, dr_leg: bool, dr_foot: bool, debug_dr: bool):
    """
    Configure DR on the unwrapped environment.
    Mask is over body_mass[1:] = [torso, thigh, leg, foot]
    torso is always fixed (mask[0]=0).
    """
    # On/off + range
    env.unwrapped.enable_dr = dr
    env.unwrapped.dr_percent = dr_percent
    env.unwrapped.debug_dr = debug_dr

    # If DR is enabled but no specific link selected -> default to all links (except torso)
    if dr and not (dr_thigh or dr_leg or dr_foot):
        env.unwrapped.dr_mask = np.array([0, 1, 1, 1], dtype=bool)
    else:
        env.unwrapped.dr_mask = np.array([0, dr_thigh, dr_leg, dr_foot], dtype=bool)


def train_and_save(
    env_id: str,
    total_timesteps: int,
    seed: int,
    lr: float,
    n_steps: int,
    batch_size: int,
    gamma: float,
    ent_coef: float,
    save_path: str,
    dr: bool,
    dr_percent: float,
    dr_thigh: bool,
    dr_leg: bool,
    dr_foot: bool,
    debug_dr: bool,
):
    env = make_env(env_id, seed)
    configure_dr(env, dr, dr_percent, dr_thigh, dr_leg, dr_foot, debug_dr)

    # Quick sanity-check: show sampled masses across resets (DON'T re-seed resets or you'll make it deterministic)
    if debug_dr:
        print("TRAIN ENV DR:", env.unwrapped.enable_dr, "dr_percent:", env.unwrapped.dr_percent, "mask:", env.unwrapped.dr_mask)
        for i in range(3):
            env.reset()  # no seed here -> should sample different params each episode if DR is on
            print("TRAIN reset", i, "masses:", env.unwrapped.get_parameters())

    policy_kwargs = dict(net_arch=[256,256])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        seed=seed,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(save_path)
    env.close()
    return save_path


def eval_model(model_path: str, env_id: str, seed: int, n_eval_episodes: int = 50):
    env = make_env(env_id, seed + 1000)

    # Evaluation should be on a fixed env (no DR during eval)
    configure_dr(
        env,
        dr=False,
        dr_percent=0.0,
        dr_thigh=False,
        dr_leg=False,
        dr_foot=False,
        debug_dr=False,
    )

    model = PPO.load(model_path)

    mean_return, std_return = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        return_episode_rewards=False,
    )
    env.close()
    return mean_return, std_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO")  # kept for clarity
    parser.add_argument("--train_env", type=str, required=True, choices=["source", "target"])
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)

    # some hyperparameters you can sweep
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent_coef", type=float, default=0.0)

    # DR flags
    parser.add_argument("--dr", action="store_true", help="Enable domain randomization")
    parser.add_argument("--dr_percent", type=float, default=0.2, help="DR range ±p (e.g., 0.2 = ±20%)")
    parser.add_argument("--dr_thigh", action="store_true", help="Randomize thigh mass")
    parser.add_argument("--dr_leg", action="store_true", help="Randomize leg mass")
    parser.add_argument("--dr_foot", action="store_true", help="Randomize foot mass")
    parser.add_argument("--debug_dr", action="store_true", help="Print masses at reset (training only)")
    args = parser.parse_args()

    train_env_id = "CustomHopper-source-v0" if args.train_env == "source" else "CustomHopper-target-v0"

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"ppo_train-{args.train_env}_seed-{args.seed}")

    print(f"\n=== Training PPO on {train_env_id} ===")
    train_and_save(
        env_id=train_env_id,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        lr=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        save_path=model_path,
        dr=args.dr,
        dr_percent=args.dr_percent,
        dr_thigh=args.dr_thigh,
        dr_leg=args.dr_leg,
        dr_foot=args.dr_foot,
        debug_dr=args.debug_dr,
    )

    # Evaluate on source and target
    for test_env, test_env_id in [
        ("source", "CustomHopper-source-v0"),
        ("target", "CustomHopper-target-v0"),
    ]:
        mean_r, std_r = eval_model(model_path, test_env_id, seed=args.seed, n_eval_episodes=50)
        print(f"Eval train({args.train_env}) -> test({test_env}): mean_return={mean_r:.2f} +/- {std_r:.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
