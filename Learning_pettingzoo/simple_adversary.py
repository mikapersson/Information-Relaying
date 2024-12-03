import numpy as np

from pettingzoo.mpe import simple_adversary_v3

#OBS the code below only works for continous_actions = True??
env = simple_adversary_v3.env(render_mode="human", N=2, max_cycles = 100, continuous_actions = True)
env.reset(seed=42)

def simple_policy(observation):
    """
    Simple policy function that chooses an action based on observation.
    In this case, it chooses a random action but could be expanded.
    """
    # For example, let's just return a random action for now - only works for continous
    return np.random.uniform(low=0, high=1, size=env.action_space(agent).shape)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()
        #action = simple_policy(observation)
        #print(action)

    env.step(action)
env.close()