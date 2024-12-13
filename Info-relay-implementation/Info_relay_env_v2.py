from pettingzoo import ParallelEnv

from gymnasium import spaces # spaces for action/observation

import numpy as np
from copy import copy

from Info_relay_classes import Drone, Base, Emitter, World

from gymnasium.utils import seeding


# class of agents (where drones and emitters(bases are included)) - boolean that shows dynamics or not - future we can have ground/air as well
# vectorized stepping function 

#maybe store all agents (drones + bases + emitters)

#fullständig info, fixed antal agenter i början - coop spel

 


##ide: ska det vara egna agent / emitter classer, eller ska allt ingå i env:et?

class Info_relay_env(ParallelEnv):
    metadata = {
        "name": "Info_relay_v2",
    }

    def __init__(self, num_agents = 10, num_bases = 2, num_emitters = 3, world_size = 100,
                 a_max = 1.0, omega_max = 1.0, step_size = 0.01, max_iter = 10):
        self.n_agents = num_agents
        self.num_bases = num_bases
        self.num_emitters = num_emitters
        self.a_max = a_max
        self.omega_max = omega_max

        self.eta = 1 # controls the probability of detecting the signal ~ threshold

        self.render_mode = None # OBS fix - or maybe have as option in reset!
        self.world_size = world_size # the world will be created as square.

        self.h = step_size
        self.max_iter = max_iter # maximum amount of iterations before the world truncates 

        self._seed() # maybe set up seed in another way?? now imprting from gymnasium

        # set all World variables first - contains all enteties
        self.world = self.make_world(num_agents, num_bases, num_emitters)


        #self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = [agent.name for agent in self.world.agents] # self.agents[:]
        
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

        self.terminated_agents = set() # keeps track of lost agents
        ## need a way to add new agents during simulation aswell!

    # these "world" functions could be included in their own class, Scenario, like in MPE -
    # - then pass Info_relay_env variables. Can be changed later on if needed/makes the program easyer  
    def make_world(self, num_agents, num_bases, num_emitters):
        """
        creates a World object that contains all enteties (drones, bases, emitters) in lists
        """
        world = World()

        world.dt = self.h ## step leangth is transferred to the WOrld

        world.dim_c = 2 # communication dimension, not exactly sure how we will use it yet

        world.agents = [Drone() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.size = 0.25 #quite small
            agent.max_speed = 1.0 # can be cnahged later! or different for differnt agents
            agent.u_range = [self.a_max, self.omega_max] # maximum control range = a_max, omega_max
            agent.internal_noise = 1 ## set internal noise level
            agent.color = np.array([0.25, 0.25, 0.25])

        world.bases = [Base() for _ in range(num_bases)]
        for i, base in enumerate(world.bases):
            base.name = f"base_{i}"
            base.size = 0.75 # the biggest pieces in the world
            base.color = np.array([0.85, 0.35, 0.35])

        world.emitters = [Emitter() for _ in range(num_emitters)]
        for i, emitter in enumerate(world.emitters):
            emitter.name = f"emitter_{i}"
            emitter.color = np.array([0.35, 0.85, 0.35])

        return world
    
    ## obs could maybe run make_world here also, otherwise potential problem with truncation ([] agent lists)
    def reset_world(self, world, np_random): # np_ranodm should be some sort of seed for reproducability
        """
        Resets the world and all enteties - called inside the reset function.
        Randomly distributes the bases and emitters - all drones start at one of the bases (first in list)
        """
        for i, base in enumerate(world.bases):
            base.state.p_pos = np_random.uniform(0, self.world_size, world.dim_p)
            base.state.p_vel = np.zeros(world.dim_p)

        for i, emitter in enumerate(world.emitters):
            emitter.state.p_pos = np_random.uniform(0, self.world_size, world.dim_p)
            emitter.state.p_vel = np.zeros(world.dim_p)

        for i, agent in enumerate(world.agents):
            agent.state.p_pos = world.bases[0].state.p_pos # see if this works - all should start at the first base
            agent.state.p_vel = np.zeros(world.dim_p) 
            agent.state.theta = 0.0 # should this just be one number? or still nparray?
        

    def reset(self, seed=None, options=None):

        if seed is not None:
            self._seed(seed=seed)
        
        self.reset_world(self.world, self.np_random)

        #if seed is not None: 
        #   np.random.seed(seed) # seed for reproducability
        
        # always start at timestep 0
        self.timestep = 0

        self.agents = copy(self.possible_agents)

        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        #self.current_actions = [None] * self.num_agents ## could maybe be used
        self.infos = {name: np.zeros(3) for name in self.agents} # for parallel env
        # currently three actions for movement - but can be expanded to include communication later on

        ## reset the action masks for the agents? - maybe only check if thay are close to the "border"?
        # implement action masks

        ## obs create observation!
        # the observation has to include the agent's position, as well as all other agents they can observe

        return None, None # while testing
   
    
    #sets the actions for each agent - another function will describe bases/emitters
    def set_action(self, action, agent):
        agent.action.u = np.zeros(3) 
        agent.action.c = np.zeros(1) ## not yes included!! expand later to incldue
                                    ### also inlcude in action_spaces!!
        if agent.movable: ## all our agents will be movable,
            agent.action.u[0] = action[0] # x velocity (or acc)
            agent.action.u[1] = action[1] # y velocity (or acc)
            agent.action.u[2] = action[2] # omega, controls theta

        ##set communication action(s) - only if to send or not to send?? direction is controlled by u[2]



    def step(self, actions):
        
        rewards = {}    
        
        #self.current_actions = actions # obs actions are a dictionary!!
        # kanske räcker att endast använda actions frä - inte self., action används endast inne i step

        #self.current_actions = {}
        #for agent in self.world.agents:
            #self.current_actions[agent] = actions.get(agent , np.zeros(3)) # default 0 if no action is given
            #self.current_actions[agent.name] = actions.get(agent.name, np.zeros(3))
        #print(self.current_actions)

        #h = self.h
        ## need to get the actions before world step!! - tgis loop fixes the action
        for agent in self.world.agents: 
            action = actions.get(agent.name, np.zeros(3)) # get the action for each agent - change 3 to num actions in the end
            #print(action)
            self.set_action(action, agent)

        self.world.step() # the world controls the motion of the agents - (also controls communication ?)
        print("world step done")

        ## here we can look at the rewards - after world step
        for agent in self.world.agents:
            rewards[agent.name] = float(self.reward(agent))

        # then make new observations and return 

        ## sen behöver även beslut om att skicka ut signaler behandlas! Riktade utsändningar!


        # handle termination - termination if agent is killed, 
        # termination for bases when x num masseges has been succesfully delivered -> terminates all
        #terminations could be stored in a self.terminations that updates each round if agnets died
        #there culd be an internal self.bases_termination that declares game end if enough masseges has gone trough
        terminations = {agent.name: False for agent in self.world.agents}

        # handle truncation - save as dict of dicts?
        #OBS - maybe truncations (and terminations/rewards) HAS to be one dict to fit with pettingzoo?
        truncations = {entity.name: False for entity in self.world.entities}
        if self.timestep > self.max_iter:
            rewards = {agent.name: 0.0 for agent in self.world.agents} # maybe add reward for bases/emitters later on? far future 
            truncations = {entity.name: True for entity in self.world.entities}
            self.agents = []
            self.bases = []
            self.emitters = []

        self.timestep += 1

        # generate new observations
        #observations = {agent.name: None for agent in self.world.agents}
        observations = self.observe_all() # call this when implemented
        print(observations)
        

        self.full_absolute_observation(self.world.agents[0])

        # render if wanted
        if self.render_mode == "human":
            self.render()

        infos = {agent.name: {} for agent in self.world.agents}

        #return observations, rewards, terminations, truncations, infos
        return observations, rewards, terminations, truncations, infos
    
    def reward(self, agent): ## OBS this could be put into the Scenario class
        """ 
        The reward given to each agent - could be made up of multiple different rewards in different fucntions
        """
        return 0.0 # just for testing
        pass


    ## some version of this could be used in the beginning so that agents do nt run away by mistake?
    def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
    

    def calculate_prob_detection(self, SNR):
        eta = self.eta 
        
    def calculate_SNR(self, agent, other):
        SNR = 0
        rel_pos = agent.state.p_pos - other.state.p_pos
        alpha = np.arctan(rel_pos[1]/rel_pos[0]) # the angle between x-axis and the line bweteen the drones
        ## if-satser för att kolla hur phi ska räknas ut
        return SNR

    def full_absolute_observation(self, agent):
        """
        Observes everything perfectly in the env, return the absolute (global) coordinates of everything
        """   
        
        # the emitters will be activated once partial observable envs are used
        # need to include some problem in observing the trasmitter
        # detection probability is a function of distance, drone antenna orientation and noise!!
        # Thus the emitters do not really need to send anything! They are observed by the agents anyway
        emitter_pos = []
        emitter_comm = []
        for emitter in self.world.emitters:
            break
        
        base_comm = []
        base_pos = []
        for base in self.world.bases:
            base_pos.append(base.state.p_pos)
            ## the probability of detecting this signal is prop to distance and noise, 
            # the direction of only the agent's antenna --> cos(\phi_r) = 1 

        other_pos = []
        other_vel = []
        other_comm = []
        for other in self.world.agents:
            if other is agent:
                continue
            else:
                other_pos.append(other.state.p_pos)
                other_vel.append(other.state.p_vel)

                #communication part
                # first check if signal is detected, then add all detected signals to the observations
                

        ## OBS really need to dubbelcheck if this is the most efficient way to store observations
        # for example: agent's own state can be stored in the same array as all others? 
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + base_pos
            + other_pos
            + other_vel
            #+ base_comm
            #+ other_comm
        )
        

    def full_relative_observation(self, agent):
        """
        Observes everything perfectly in the env, return the relative coordinates of everything
        """   
        pass

    def partial_observation(self, agent):
        """
        Here the partial observation model will be implemenetd later on
        """
        pass
    
    def observe(self, agent): # these could be included in the Scenario class together with world make/reset
        #now only return the state of all entities as one big np array
        return self.full_absolute_observation(agent)

    def observe_all(self):
        """Return observations for all agents as a dictionary"""
        return {agent.name: self.observe(agent) for agent in self.world.agents}


    def render(self):
        pass # try to import something already done?
            # or maybe use basic graphics in python for simplicity - later unity 


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    # OBS maybe add @functools.lru_cache(maxsize=None) - could reduce compute time
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]