from pettingzoo import ParallelEnv

from gymnasium import spaces # spaces for action/observation

import numpy as np
from copy import copy



##ide: ska det vara egna agent / emitter classer, eller ska allt ingå i env:et?

class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, num_agents = 10, num_bases = 2, num_emitters = 3, world_size = 100,
                 a_max = 1, omega_max = 1):
        self.num_agents = num_agents
        self.num_bases = num_bases
        self.num_emitters = num_emitters
        self.a_max = a_max
        self.omega_max = omega_max

        # these three following inits maybe should be in reset() instead?
        self.possible_agents = [f"agent_{index}" for index in range(self.num_agents)]
        self.possible_bases = [f"base_{index}" for index in range(self.num_bases)]
        self.possible_emitters = [f"emitter_{index}" for index in range(self.num_emitters)]
        
        #all agents get their own obs/action spaces - stored in a Dict structure 
        #obs bases should observe for masseges aswell, right? They can use the same obs_space with different masking 
        self.observation_spaces = spaces.Dict(
            {agent: spaces.Box(low = 0.0, high = 1.0, 
            shape = self.num_agents + self.num_bases + self.num_emitters, dtype = np.float64) 
            for agent in self.possible_agents} ## kanske BOX(num total agents+other)?
        ) #OBS!! fixa till och dubbelkolla - behöver även kunna observera meddelanden? Hur kodas?

        self.action_spaces = spaces.Dict({agent: spaces.Box(
            low = np.array([-self.a_max, -self.a_max, -self.omega_max]), 
            high = np.array([self.a_max, self.a_max, self.omega_max]), dtype=np.float64) 
            for agent in self.possible_agents}) # here we have three actions, acceleration in x/y and rotation

        #self.agent_pos = []
        self.bases_pos = []
        self.emitter_pos = []

        #self.agent_vel = []
        self.agent_coords = {} #contains pos, vel and theta

        self.timestep = None
        self.render_mode = None # OBS fix - or maybe have as option in reset!
        self.world_size = world_size # the world will be created as square.

        self.terminated_agents = set() # keeps track of lost agents
        ## need a way to add new agents during simulation aswell!

    def reset(self, seed=None, options=None):
        if seed is not None: 
            np.random.seed(seed) # seed for reproducability
        
        self.timestep = 0

        # init pos for emitters/agents (emitters get random pos in the playing area)

        self.agents = copy(self.possible_agents)
        self.bases = copy(self.possible_bases)
        self.emitters = copy(self.possible_emitters)
        
        self.bases_pos = [(np.random.uniform(0, self.world_size), np.random.uniform(0, self.world_size))
                          for _ in range(self.num_bases)] # randomly place the bases in the playing field
        
        self.emitters_pos = [(np.random.uniform(0, self.world_size), np.random.uniform(0, self.world_size))
                          for _ in range(self.num_emitters)]
        
        #currently all agents start at the same base (and coordinate) - no collisions 
        #self.agent_pos = [self.bases_pos[0] for _ in range(self.num_agents)]
        #self.agent_vel = [(0, 0) for _ in range(self.num_agents)]
        #self.agent_theta = [0 for _ in range(self.num_agents)]
        self.agent_coords = {
            "pos": [self.bases_pos[0] for _ in range(self.num_agents)],
            "vel": [(0, 0) for _ in range(self.num_agents)],
            "theta": [0 for _ in range(self.num_agents)]
        }

        ## reset the action masks for the agents? - maybe only check if thay are close to the "border"?
        # implement action masks

        ## obs create observation!
        # the observation has to include the agent's position, as well as all other agents they can observe

    def step(self, actions):
        pass

        # first move all agents (and later bases/emitters) based on their velocities
        agent_actions = {}
        for i in range(self.num_agents):
            agent_actions[f"agent_{i}"] = actions.get(f"agent_{i}", np.zeros(3)) # default 0 if no action is given
        
        ##obs the actions are accelerations in different directions (and rotational acceleration)
        for i in range(self.num_agents):
            pass

        #### notering -- koda in de härledda uttrycken som finns angivna i overleafen!!


        # handle termination

        # handle truncation

        # generate new observations

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        pass # try to import something already done?
            # or maybe use basic graphics in python for simplicity - later unity 

    # OBS maybe add @functools.lru_cache(maxsize=None) - could reduce compute time
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]