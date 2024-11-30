import time
import numpy as np

#the first env is a simpler version with only one prisoner, the second is largely the same but ith multiple prisoners
#from custom_env_tutorial import CustomActionMaskedEnvironment
from custom_env_multiple_agents import CustomActionMaskedEnvironment

parallel_env = CustomActionMaskedEnvironment(render_mode="human")
observations, infos = parallel_env.reset(seed=1, options = [7])


def policy(agent,observations):
    #observation = observations[agent]['observation'] # can be used later for (ex) evading gaurd 
    action_mask = observations[agent]['action_mask']

    print(action_mask)
    available_actions = np.where(action_mask == 1)[0] # used now when action_mask is only np.array
    #available_actions = [i for i, mask in enumerate(action_mask) if mask == 1] # used before when action_mask was a list or array
    print(available_actions) 
    return np.random.choice(available_actions)

while parallel_env.agents:
    # this is where you would insert your policy
    #actions = {agent: parallel_env.action_space(agent).sample() for agent in parallel_env.agents} always random

    actions = {}
    for agent in parallel_env.agents:
        action = policy(agent, observations) # the policy can be different for dfferent type of agents

        actions[agent] = action

    observations, rewards, terminations, truncations, infos = parallel_env.step(actions)
    if not parallel_env.agents:
        print("termination: ", terminations)
        print("truncation: ", truncations)
        print("rewards: ", rewards)

    time.sleep(0.1) # to see the game play out
parallel_env.close()