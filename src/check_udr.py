import gymnasium as gym
from env.custom_hopper import *

env = gym.make("CustomHopper-source-v0", domain_randomization=True, dr_percent=0.2)

print("domain_randomization:", env.unwrapped.domain_randomization)
print("dr_percent:", env.unwrapped.dr_percent)

for ep in range(5):
    env.reset(seed=ep)
    print(f"Episode {ep} masses:", env.unwrapped.get_parameters())

env.close()