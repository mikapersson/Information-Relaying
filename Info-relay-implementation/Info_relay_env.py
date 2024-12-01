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
                 a_max = 1, omega_max = 1, step_size = 0.01):
        self.n_agents = num_agents
        self.num_bases = num_bases
        self.num_emitters = num_emitters
        self.a_max = a_max
        self.omega_max = omega_max


        self.h = step_size
        self.max_iter = 10 # change 


        # these three following inits maybe should be in reset() instead?
        self.possible_agents = [f"agent_{index}" for index in range(self.n_agents)]
        self.possible_bases = [f"base_{index}" for index in range(self.num_bases)]
        self.possible_emitters = [f"emitter_{index}" for index in range(self.num_emitters)]
        
        #all agents get their own obs/action spaces - stored in a Dict structure 
        #obs bases should observe for masseges aswell, right? They can use the same obs_space with different masking 
        self.observation_spaces = spaces.Dict(
            {agent: spaces.Box(low = 0.0, high = 1.0, 
            shape = (self.n_agents + self.num_bases + self.num_emitters,), dtype = np.float64) 
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
        """
        self.agent_coords = {
            "pos": [self.bases_pos[0] for _ in range(self.n_agents)],
            "vel": [(0, 0) for _ in range(self.n_agents)],
            "theta": [0 for _ in range(self.n_agents)]
        }
        """
        #OBS: kan formulra dicten på annat sätt istället:
        #self.agent_coords = {agent: (...) for agent in self.agents}
        #här kan (...) t.ex. vara en nparray med fem entries

        self.agent_coords = {agent: np.array([self.bases_pos[0][0], self.bases_pos[0][1], 
                                             0, 0, 0]) for agent in self.agents}

        ## reset the action masks for the agents? - maybe only check if thay are close to the "border"?
        # implement action masks

        ## obs create observation!
        # the observation has to include the agent's position, as well as all other agents they can observe

        return None, None # while testing

    def step(self, actions):
        
        rewards = {}    

        h = self.h

        # first move all agents (and later bases/emitters) based on their velocities
        agent_actions = {}
        for i in range(self.n_agents):
            agent_actions[f"agent_{i}"] = actions.get(f"agent_{i}", np.zeros(3)) # default 0 if no action is given
        print(agent_actions)
        ##obs the actions are accelerations in different directions (and rotational acceleration)
        for i in range(self.n_agents):
            agent_coords = self.agent_coords[f"agent_{i}"]
            agent_action = agent_actions[f"agent_{i}"]
            print(self.agent_coords[f"agent_{i}"])
            self.agent_coords[f"agent_{i}"][0] += h * agent_coords[0] + h**2/2 * agent_action[0] # x_pos
            self.agent_coords[f"agent_{i}"][1] += h * agent_coords[1] + h**2/2 * agent_action[1] # y_pos
            self.agent_coords[f"agent_{i}"][2] += h * agent_action[0] # x_vel 
            self.agent_coords[f"agent_{i}"][3] += h * agent_action[1] # y_vel
            self.agent_coords[f"agent_{i}"][4] += h * agent_action[2] # theta
            print(self.agent_coords[f"agent_{i}"])

        ##OBS - behöver också koda in Sigma matrisen!

        ## sen behöver även beslut om att skicka ut signaler behandlas! Riktade utsändningar!


        # handle termination - termination if agent is killed, 
        # termination for bases when x num masseges has been succesfully delivered -> terminates all

        # handle truncation - save as dict of dicts?
        #OBS - maybe truncations (and terminations/rewards) HAS to be one dict to fit with pettingzoo?
        truncations = {}
        truncations["agents"] = {agent: False for agent in self.agents}
        truncations["bases"] = {base: False for base in self.bases}
        truncations["emitters"] = {emitter: False for emitter in self.emitters}
        if self.timestep > self.max_iter:
            rewards["agents"] = {agent: 0 for agent in self.agents} # maybe add reward for bases/emitters later on 
            # now rewards is dict of dicts - can be just dict as only agents get rewards at this stage
            truncations["agents"] = {agent: True for agent in self.agents}
            truncations["bases"] = {base: True for base in self.bases}
            truncations["emitters"] = {emitter: True for emitter in self.emitters}
            self.agents = []
            self.bases = []
            self.emitters = []


        # generate new observations

        if self.render_mode == "human":
            self.render()

        #return observations, rewards, terminations, truncations, infos
        return None, None, None, truncations, None
    
    def observe(self):
        pass

    def observe_all(self):
        """Return observations for all agents."""
        return {agent: self.observe(agent) for agent in self.agents}


    def render(self):
        pass # try to import something already done?
            # or maybe use basic graphics in python for simplicity - later unity 

    # OBS maybe add @functools.lru_cache(maxsize=None) - could reduce compute time
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]