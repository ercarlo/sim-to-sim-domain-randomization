"""Test a random policy on the Gym Hopper environment

    Play around with this code to get familiar with the
    Hopper environment.

    For example, what happens if you don't reset the environment
    even after the episode is over?
    When exactly is the episode over?
    What is an action here?
"""
import gymnasium as gym
from env.custom_hopper import *

def main():
    render = True

    if render:
        env = gym.make('CustomHopper-source-v0', render_mode='human')
    else:
        env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    print('State space:', env.observation_space)  # state-space
    print('Action space:', env.action_space)  # action-space
    print('Dynamics parameters:', env.unwrapped.get_parameters())  # masses of each link of the Hopper

    n_episodes = 500

    for ep in range(n_episodes):  
        done = False
        state, info = env.reset()  # Reset environment to initial state

        while not done:  # Until the episode is over
            action = env.action_space.sample()  # Sample random action

            state, reward, terminated, truncated, _ = env.step(action)  # Step the simulator to the next timestep
            done = terminated or truncated

            if render:
                env.render()


if __name__ == '__main__':
    main()