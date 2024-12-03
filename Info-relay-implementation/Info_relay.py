import time
import numpy as np

#the first env is a simpler version with only one prisoner, the second is largely the same but ith multiple prisoners
#from custom_env_tutorial import CustomActionMaskedEnvironment
from Info_relay_env import CustomEnvironment

parallel_env = CustomEnvironment(num_agents=2)
print("init done")
observations, infos = parallel_env.reset(seed=42)
print("reset done")

def policy(agent,observations): 
    #return np.random.uniform(low=0, high=1, size=parallel_env.action_space(agent).shape)
    return np.ones(parallel_env.action_space(agent).shape)

while parallel_env.agents:
    # this is where you would insert your policy
    actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents} #always random
    """
    actions = {}
    for agent in parallel_env.agents:
        action = policy(agent, observations) # the policy can be different for dfferent type of agents

        print(action)

        actions[agent] = action
    """
    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)

    if not parallel_env.agents:
        print("termination: ", terminations)
        print("truncation: ", truncations)
        print("rewards: ", rewards)

    time.sleep(0.1) # to see the game play out

    #break
parallel_env.close()