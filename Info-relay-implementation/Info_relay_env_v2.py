from pettingzoo import ParallelEnv

import gymnasium

from gymnasium import spaces # spaces for action/observation

import numpy as np
from copy import copy, deepcopy
import random

# to be able to import from current directory in both linux and windows
try:
    from Info_relay_classes import Drone, Base, Emitter, World#, Message  
except ImportError:
    from .Info_relay_classes import Drone, Base, Emitter, World#, Message

from gymnasium.utils import seeding

# to render basic graphics
import pygame

import itertools


# class of agents (where drones and emitters(bases are included)) - boolean that shows dynamics or not - future we can have ground/air as well
# vectorized stepping function 

# maybe store all agents (drones + bases + emitters)

# fullständig info, fixed antal agenter i början - coop spel

 
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

##ide: ska det vara egna agent / emitter classer, eller ska allt ingå i env:et?


# BIG TODO (optional) - flytta ut, make_world, reset_world, observe, reward, terminate med hjälpfunktioner till egna classer -
# - en klass per scenario. Detta liknar det som är byggt i pettingzoo MPE environment. 
# likt hur det är kodat i mapparna under https://github.com/Farama-Foundation/PettingZoo/tree/master/pettingzoo/mpe 

# TODO(SAMSAM) - lägg till fiende och störsändningsfunktionalitet. Fiendens state (position) ska uppdateras i info_relay_classes likt agenterna.
# TODO(SAMSAM?) - baserna börjar endast på x-axeln. Samma bas sänder varje spel - börjar alltid i origo

# TODO - Adam har funderingar kring att "få en känsla kring valet av parametrar och slumpning av begynnelsevärde", där ingår att han vill
# implementera en baseline.

# TODO - dubbelkolla belöningsfunktionen

# TODO - Enable gpu träning ? Kanske inte behövs

# TODO (optional) - Låt agenter dela med avsikt till andra agenter. 

# TODO (optional & low prio) - Beslut om position snarare än färdriktning

class Info_relay_env(ParallelEnv):
    metadata = {
        "name": "Info_relay_v2",
        "render_fps": 1
    }

    def __init__(self, num_agents = 1, num_bases = 2, num_emitters = 1, world_size = 1,
                 a_max = 1.0, omega_max = np.pi/4, step_size = 0.05, max_cycles = 25, 
                 continuous_actions = True, one_hot_vector = False, antenna_used = True, 
                 com_used = True, num_messages = 1, base_always_transmitting = True, 
                 random_base_pose = True, observe_self = True, render_mode = None):
        #super().__init__()
        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = 700
        self.height = 700
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1
        self.game_font = pygame.freetype.SysFont('Arial', 24)

        self.renderOn = False

        self.n_agents = num_agents
        self.num_bases = num_bases
        self.num_emitters = num_emitters
        self.a_max = a_max
        self.omega_max = omega_max

        self.SNR_threshold = 1 # threshold for signal detection

        self.world_size = world_size # the world will be created as square. - maybe not used now

        self.h = step_size
        self.max_iter = max_cycles # maximum amount of iterations before the world truncates - OBS renamed the inupt to more closely match benchmarl

        self._seed() # maybe set up seed in another way?? now imprting from gymnasium

        self.num_messages = num_messages # to keep track of the number of messages = to the number of message buffer slots the agents will have

        # set all World variables first - contains all enteties
        self.world = self.make_world(num_agents, num_bases, num_emitters)

        self.continuous_actions = continuous_actions # indicies if continous or discrete actions are used
        self.one_hot_vector = one_hot_vector # if continous - this shows if one hot vector or single value representation of actions (output from NN) are used

        self.antenna_used = antenna_used #OBS ändra när antennen ska styras igen
        self.com_used = com_used #OBS - communications can be turned of to test just the movement in different tasks
        self.observe_self = observe_self
        self.angle_coord_rotation = 0 # declared as a class variabel to be reach in observation and in setting actions

        self.base_always_transmitting = base_always_transmitting # decides if the base is sending every time step. If false, the base send sporadically

        self.random_base_pose = random_base_pose # if the bases starting positions are random or always located on the x-axis

        self.possible_agents = [agent.name for agent in self.world.agents] 


        dim_p = self.world.dim_p
        
        # Ny kommentar: används ej inne i klassen - vet ej om benchmarl letar efter detta så avvaktar med att ta bort till testat med benchmarl
        # OBS really need to dubble check the state space - what is to be included - is it neccessary??
        state_dim = dim_p * (self.n_agents - 1) + dim_p * self.num_bases + self.n_agents
        self.state_space = spaces.Box(
            low=-np.float32(np.inf),
            high=+np.float32(np.inf),
            shape=(state_dim,),
            dtype=np.float32,
        )
        
        if self.antenna_used:
            self.observation_spaces = {
                agent.name: spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.n_agents + dim_p * (self.n_agents - 1 + observe_self) + dim_p * self.num_bases + self.n_agents,),
                    dtype=np.float32
                ) for agent in self.world.agents
            }
        else:
            self.observation_spaces = {
                agent.name: spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(dim_p * (self.n_agents - 1 + observe_self) + dim_p * self.num_bases + self.n_agents,),
                    dtype=np.float32
                ) for agent in self.world.agents
            }

        # sets the action spaces for the two different continous action space cases
        if self.continuous_actions:
            if self.one_hot_vector: # inte updaterad
                self.action_spaces = {
                agent.name: spaces.Box(
                    low=np.concatenate((
                        np.array([0.0, 0.0, 0.0]),  
                        np.zeros(2 * (agent.message_buffer_size + 1))  
                    )),
                    high=np.concatenate((
                        np.array([1.0, 1.0, 1.0]),  
                        np.ones(2 * (agent.message_buffer_size + 1))  
                    )),
                    dtype=np.float64
                ) 
                for agent in self.world.agents
            }

            else:
                self.action_spaces = {
                agent.name: spaces.Box(
                    low=np.array([0.0, 0.0, 0.0]),  
                    high=np.array([1.0, 1.0, 1.0]),  
                    dtype=np.float64
                ) 
                for agent in self.world.agents
                }

        else:
            # chose the number of discretized actions (ex velocity: [-max, 0, +max])
            num_velocity_actions = 3
            num_angle_actions = 3 # should be atleast 3 (and uneven) if used
            
            if not self.antenna_used: # the agents do not use the antenna - isotropic transmission
                num_angle_actions = 1
            
            # Here we combine the different actions into one single output - many actions that have to be 'decoded'
            # used later to retrieve the correct action index for each major action type
            
            self.different_discrete_actions = [num_velocity_actions, num_velocity_actions,
                                               num_angle_actions]
            
            self.action_mapping_dict = {i: list(comb) for i, comb in enumerate(itertools.product(*map(range, self.different_discrete_actions)))}
            
            self.action_spaces = {agent.name: spaces.Discrete(
                num_velocity_actions**2 * num_angle_actions
            ) for agent in self.world.agents}

        self.transmission_radius_bases = self.calculate_transmission_radius(self.world.bases[0])
        self.transmission_radius_drones = self.calculate_transmission_radius(self.world.agents[0])

        self.recived_messages_bases = [] # an attribute that keeps track of all messages recieved by bases THIS timestep    

        self.episode_counter = 0 # checks how many times the environemnt has been reset. Used for continously changing the starting states


    # these "world" functions could be included in their own class, Scenario, like in MPE -
    # - then pass Info_relay_env variables. Can be changed later on if needed/makes the program easyer  
    def make_world(self, num_agents, num_bases, num_emitters):
        """
        creates a World object that contains all enteties (drones, bases, emitters) in lists
        """
        world = World()

        world.dt = self.h ## step length is transferred to the World

        world.dim_c = 2 # communication dimension, not exactly sure how we will use it yet

        world.agents = [Drone() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.size = 0.025/2 #quite small
            agent.max_speed = 1.0 # can be cnahged later! or different for differnt agents
            agent.u_range = [self.a_max, self.omega_max] # maximum control range = a_max, omega_max
            agent.internal_noise = 1 ## set internal noise level
            agent.color = np.array([0, 0.33, 0.67])
            agent.message_buffer_size = self.num_messages

            #agent.transmit_power = 1 # set to reasonable levels

        world.emitters = [Emitter() for _ in range(num_emitters)]
        for i, emitter in enumerate(world.emitters):
            emitter.name = f"jammer_{i}"
            emitter.size = 0.025/2
            emitter.max_speed = 1.0
            agent.u_range = [self.a_max, self.omega_max] # maximum control range = a_max, omega_max
            emitter.internal_noise = 1
            emitter.color = np.array([1.0, 0, 0])            


        world.bases = [Base() for _ in range(num_bases)]
        for i, base in enumerate(world.bases):
            base.name = f"base_{i}"
            base.size = 0.050/2 # the biggest pieces in the world
            base.color = np.array([0.35, 0.85, 0.83])
            #base.transmit_power = 1 # set to reasonable levels

        #world.bases[1].silent = True # first scenario is one way communication
        #world.bases[0].generate_messages = False # the same message all the time


        return world

    # TODO - remove
    def generate_base_positions(self, np_random, radius):
        """Generate random base positions with equal spacing.
        Used in reset_world to place bases."""
        # Generate evenly spaced points on a circle
        angles = np.linspace(0, 2 * np.pi, self.num_bases, endpoint=False)

        if self.random_base_pose:
        # Randomly rotate the formation
            random_rotation = np_random.uniform(0, 2 * np.pi)
            angles += random_rotation
        
        positions = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

        # Randomly shift within bounds
        #shift = np_random.uniform(-self.world_size/2, self.world_size/2, size=(1, 2))
        #positions += shift

        return positions  
        
    # TODO - new generate base pos
    # def generate_base_positions(self, np_random, radius):
    #     """Generate random base positions with a certain distance.
    #     The bases are always located on the x-axis, with the transmitting base at (0,0)
    #     Used in reset_world to place bases."""
        
    #     x_positions = np.linspace(0,radius,2)
    #     y_positions = np.zeros(2)
    #     print(y_positions)
    #     print(radius)
    #     print(positions)
    #     return positions   
    
    
    def generate_entity_positions(self, np_random, base_positions, radius, n_entities):
        """
        Generate positions for agents inside a circular disk with a certain radius. 
        The center of the disk is the the middlepoint of the bases.  
        """
        positions = []
    
        self.center = np.mean(base_positions, axis=0)  # Midpoint of bases
        for i in range(n_entities):
            radius_agent = np_random.uniform(0, radius)  # Random radius
            angle = np_random.uniform(0, 2 * np.pi)  # Random angle
            offset = np.array([radius_agent * np.cos(angle), radius_agent * np.sin(angle)])  # Convert to Cartesian
            positions.append(self.center + offset)

        return np.array(positions)
    
    def reset_world(self, world, np_random): # np_ranodm should be some sort of seed for reproducability
        """
        Resets the world and all enteties - called inside the reset function.
        Randomly distributes the bases and emitters - all drones start at one of the bases (first in list)
        """
        for i, emitter in enumerate(world.emitters):
            #emitter.state.p_pos = np_random.uniform(-self.world_size, self.world_size, world.dim_p)
            emitter.state.p_pos = np.zeros(world.dim_p)
            emitter.state.p_vel = np.zeros(world.dim_p)

        world.message_ID_counter = 0 # resets the message_ID counter
        self.radius = self.calculate_transmission_radius(self.world.bases[0]) * (self.n_agents + 1)/2 
        if self.num_bases == 3:
            self.radius = self.calculate_transmission_radius(self.world.bases[0]) * (2 + 1)/2 # always the same distance as 2 agents - does not work otherwise

        #min_base_radius = min(max(1/self.n_agents, self.episode_counter / 5000), 0.9) 
        #min_base_radius = 1/self.n_agents
        #base_positions = self.generate_base_positions(np_random, self.radius * np_random.uniform(min_base_radius, 1.0))
        base_positions = self.generate_base_positions(np_random, self.radius * np_random.uniform(0.9, 1.0))
        #base_positions = self.generate_base_positions(np_random, self.radius * np_random.uniform(0.8, 1.0))
        for i, base in enumerate(world.bases):
            #base.state.p_pos = np_random.uniform(-self.world_size, self.world_size, world.dim_p)
            base.state.p_pos = base_positions[i]
            base.state.p_vel = np.zeros(world.dim_p)

            base.color_intensity = 0 # resets graphics so it does not look like it is sending

        world.R = np.linalg.norm(world.bases[0].state.p_pos - world.bases[1].state.p_pos)

        world.bases[0].state.c = 1
        world.bases[1].state.c = 0
        
        world.bases[1].silent = True # only one base sending
        world.bases[0].generate_messages = False # the same message all the time
        world.bases[0].silent = False 
            

        # Compute the midpoint of all bases
        #base_positions = np.array(positions)
        self.center = np.mean(base_positions, axis=0)  # Midpoint of bases

        # I feel lie the center point should be contained in world but idk
        world.center = self.center

        #radius = self.radius * min(self.episode_counter / 2500, 1) # increases from 0 to 1 
        #radius = self.radius*2
        radius = self.radius*1.5
        #radius = self.radius

        agent_positions = self.generate_entity_positions(np_random, base_positions, radius, self.n_agents)

        for i, agent in enumerate(world.agents):
            #agent.state.p_pos = np.array(world.bases[0].state.p_pos) # all starts at the first base
            #agent.state.p_pos = np_random.uniform(-self.world_size*3, self.world_size*3, world.dim_p) # randomly assign starting location in a square
            agent.state.p_pos = agent_positions[i]
            agent.state.p_vel = np.zeros(world.dim_p) 
            agent.state.c = 0 # ingen agent börjar med meddelande - alltså sänder ingen i början - kanske inte behöver denna - kör bara .message_buffer

            agent.state.theta = np.random.uniform(0,2*np.pi)
            # initiate the message_buffer so that it always has the same size
            agent.message_buffer = False

        emitter_positions = self.generate_entity_positions(np_random, base_positions, radius, self.num_emitters)

        for i, emitter in enumerate(world.emitters):
            #agent.state.p_pos = np.array(world.bases[0].state.p_pos) # all starts at the first base
            #agent.state.p_pos = np_random.uniform(-self.world_size*3, self.world_size*3, world.dim_p) # randomly assign starting location in a square
            emitter.state.p_pos = emitter_positions[i]
            emitter.state.p_vel = np.zeros(world.dim_p) 

            emitter.state.theta = 0.0 

    def reset(self, seed=None, options=None): # options is dictionary
        
        #if options is not None:
        #    self.render_mode = options["render_mode"]
        
        if seed is not None:
            self._seed(seed=seed)
        
        self.reset_world(self.world, self.np_random)

        self.episode_counter += 1 # update the iteration counter - number of reseted envs
        
        # always start at timestep 0
        self.timestep = 0 

        self.agents = copy(self.possible_agents) #OBS används denna fortfarande? 

        observations = self.observe_all()
        
        if not self.continuous_actions: # Nu kör vi utan action masking - finns inga otillåtna actions när agenterna inte beslutar om sändning
            infos = {agent.name: {"action_mask" : np.ones(len(self.action_mapping_dict), dtype=int)} for agent in self.world.agents}
        elif self.continuous_actions:
            infos = {agent.name: None for agent in self.world.agents}
        else:
            infos = {agent.name: None for agent in self.world.agents} # this case (one discrete action) is probably not used but is nedded to be changed in that case

        return observations, infos 
    

    #sets the action in the case where all actions are discrete
    # OBS: the self.a_max variable can be changed to an agent-specific value to keep track of 
    def set_discrete_action(self, action, agent):
        agent.action.u = np.zeros(3)
        
        # here action is a number between 0 and 143 (the number of combinations of all actions) - translate it into the different subactions
        # the for loop below ensures unique mapping between action number and a set of subactions 
        actions = self.action_mapping_dict.get(action)
        agent.action.u[0] = actions[0]*self.a_max # actions is currently [0,1,2] 
        if actions[0] == 2:
            agent.action.u[0] = -self.a_max
        agent.action.u[1] = actions[1]*self.a_max
        if actions[1] == 2:
            agent.action.u[1] = -self.a_max
        
        if self.antenna_used: # the agents control their antenna - do not send isotropical (otherwise action always 0)
            agent.action.u[2] = actions[2]*self.omega_max
            if actions[2] == 2:
                agent.action.u[2] = -self.omega_max


    # sets action when all outputs are continuous
    def set_continuous_action(self, action, agent):

        # the one hot vector approch will probably not be used 
        def set_com_action_one_hot_vector(action, agent):
            buffer_size = agent.message_buffer_size + 1

            agent.action.c = np.argmax(action[3 : 3 + buffer_size])  

        def set_com_action_one_output(action, agent):
            agent.action.c = round(action[3] * agent.message_buffer_size) # translates to integer

        agent.action.u = np.zeros(3)
        agent.action.c = 0

        # 2x-1 transforms the input of [0,1] to [-1,1] (the NN outputs a number \in [0,1])
        agent.action.u[0] = (2*action[0]-1)*self.a_max # x velocity (or acc)
        agent.action.u[1] = (2*action[1]-1)*self.a_max # y velocity (or acc)
        agent.action.u[2] = (2*action[2]-1)*self.omega_max # omega, controls theta

        # set com action
        if self.one_hot_vector:
            set_com_action_one_hot_vector(action, agent)
        else:
            set_com_action_one_output(action, agent)


    def set_action(self, action, agent):
        if self.continuous_actions:
            self.set_continuous_action(action, agent)
        else:
            self.set_discrete_action(action, agent)
    

    def step(self, actions):
       
        rewards = {}    
        
        for agent in self.world.agents: 
            action = actions.get(agent.name, np.zeros(5)) # get the action for each agent
            self.set_action(action, agent)

    
        #print("step", self.timestep)
        self.world.step() # the world controls the motion of the agents - (also controls communication ?)
        #print("world step done")

        # run all comunications in the env
        self.communication_kernel()

        ## here we can look at the rewards - after world step - could be done after observations instead!
        global_reward = self.global_reward()
        reward_help = {agent.name: 0 for agent in self.world.agents} #self.transmission_reward()
        for agent in self.world.agents:
            rewards[agent.name] = float(self.reward(agent, global_reward, reward_help[agent.name]))
        
        terminations = self.terminate()
        #terminations = {agent.name: False for agent in self.world.agents}

        # handle truncation
        truncations = {agent.name: False for agent in self.world.agents}
        if self.timestep > self.max_iter - 2:
            #rewards = {agent.name: 0.0 for agent in self.world.agents} # maybe add reward for bases/emitters later on? far future 
            truncations = {agent.name: True for agent in self.world.agents}
            self.agents = []

        self.timestep += 1

        # generate new observations
        observations = self.observe_all()

        # render if wanted
        if self.render_mode == "human":
            self.render()

        infos = {agent.name: {"action_mask" : np.ones(len(self.action_mapping_dict), dtype=int)} for agent in self.world.agents}

        return observations, rewards, terminations, truncations, infos
    
    # TODO - om ny separat wrapper class för varje scenario skapas ska denna ligga där också
    def terminate(self):
        """
        Determins the termination condition for each scenario. 
        Returns a dict of the terminated agents.

        Scenario 1: terminates after first correct message delivered
        """
        if self.world.bases[1].message_buffer:
            return {agent.name: True for agent in self.world.agents}

        return {agent.name: False for agent in self.world.agents}


    def calculate_action_penalties(self, agent):
        """
        Calculate the penalties each drone recieves for actions. Movement and communication.
        """
        penalties = 0
        if self.continuous_actions:
            penalties += abs(agent.action.u[0]**2*agent.movement_cost)
            penalties += abs(agent.action.u[1]**2*agent.movement_cost)
            penalties += abs(agent.action.u[2]**2*agent.radar_cost)
        else:
            if abs(agent.action.u[0]) == abs(agent.action.u[1]): # both are 0 or max_vel
                penalties += abs(agent.action.u[0]**2*agent.movement_cost)
            else:
                penalties += abs(agent.action.u[0]**2*agent.movement_cost)
                penalties += abs(agent.action.u[1]**2*agent.movement_cost)
            penalties += abs(agent.action.u[2]**2*agent.radar_cost)


        # Communication costs - TODO någon mening att ha kvar kostnad om agenten inte själb beslutar om att sända?
        if not agent.action.c == 0: # sending
            penalties += abs(agent.transmission_cost)

        return penalties
    
    # 2_way_update - 25 points reward for the first messages delivered to each base - 50 once both are done
    def global_reward(self):
        """
        Rewards given to all agents. Given by correctly delivering messages.
        """
        reward = 0

        if self.world.bases[1].message_buffer: # meddelandet har levererats (detta tidsteg)
            pass # TODO lägg in korrekt funktion här

        return reward

    def bound(self, agent):
        """
        Computes a penalty if the agent is outside the allowed area.
        This assumes a circular game area

        Parameters:
            agent_pos (np.ndarray): Agent's current position.
            center (np.ndarray): Center of the spawn disk.
            radius (float): Radius of the allowed spawn area.

        Returns:
            float: Penalty value.
        """
        dist = np.linalg.norm(agent.state.p_pos - self.center)
        normalized_dist = dist / self.radius  # Normalize relative to allowed radius

        if normalized_dist < 1.2:
            return 0
        if normalized_dist < 1.3:
            return (normalized_dist - 0.9) * 10
        return min(np.exp(2 * normalized_dist - 2), 10)

    
    def reward(self, agent, global_reward, reward_help): ## OBS this could be put into the Scenario class
        """ 
        The reward given to each agent - could be made up of multiple different rewards in different functions
        """
        # TODO - update to new reward
        return global_reward - self.calculate_action_penalties(agent)*10 #+ reward_help/10
    
    def get_entity_by_name(self, name):
        """
        Returns the entity instance correpsonding to a name
        """
        for entity in self.world.agents + self.world.bases + self.world.emitters:
            if entity.name == name:
                return entity
        
        return None ## The entity does not exist

    
    def communication_kernel(self):
        """ Runs the communication kernel when the message buffer is just one boolean contaning one message without any meta data """
        # om bas-looparna tas bort:
        #base1 = self.world.bases[0]
        #base2 = self.world.bases[1]
        # decide if the base is sending

        self.check_base_com()

        for agent in self.world.agents:
            if agent.message_buffer: # if the agent already has the message
                continue

            recieved_message = False

            for base in self.world.bases: # really dont need to loop through the bases as only one base is sending now - only check the first base
                if base.state.c == 1: # if sending
                    SNR = self.calculate_SNR(agent, base, self.world.emitters)
                    if self.check_signal_detection(SNR):
                        agent.message_buffer = True
                        agent.state.c = 1 # not it will send continously 
                        recieved_message = True

            if recieved_message: # do not check agents if message recieved from base
                continue

            for other in self.world.agents: # the agents are always transmitting
                if other.name == agent.name:
                    continue
                if other.message_buffer:
                    SNR = self.calculate_SNR(agent, other, self.world.emitters)
                    if self.check_signal_detection(SNR):
                        agent.message_buffer = True
                        agent.state.c = 1
                        continue

        for base in self.world.bases: # maybe remove loop - only look at the 2nd base
            for agent in self.world.agents:
                SNR = self.calculate_SNR(base, agent, self.world.emitters)
                if self.check_signal_detection(SNR):
                    base.message_buffer = True # the game should end once this condition is met
                    continue
            for other in self.world.bases: # maybe remove loop - only look at the first base
                if other.name == base.name:
                    continue
                SNR = self.calculate_SNR(base, other, self.world.emitters)
                if self.check_signal_detection(SNR):
                    base.message_buffer = True
                    continue

    
    def check_base_com(self):
        """ Checks if the base should send this timestep """
        if not self.base_always_transmitting:
            self.world.bases[0].state.c = np.random.binomial(1, 0.2) # singlar slant om den ska sända eller ej
 

    def calculate_transmission_radius(self, entity):
        """
        Calculates the transmission distance that bases can transmit (signal strong enough to detect) - in order to plot 
        """
        return np.sqrt(entity.transmit_power/self.SNR_threshold)

    def check_signal_detection(self, SNR): # detection or not - based on SNR
        #SNR = self.calculate_SNR(agent, other)
        if SNR > self.SNR_threshold:
            return True # signal detected
        else:
            return False # signal not detected


    def calculate_SNR(self, reciever, transmitter, jammers):
        """
        Calculates the SNR between transmitter and reciever. It is assumed that all entities can listen uniformly in all directions.
        All bases send uniformly in all directions. All agents send in the direction of its antenna
        """
        if self.antenna_used:
            # agent is reciever, r, other is transmitter, t
            SNR = 0
            rel_pos = transmitter.state.p_pos - reciever.state.p_pos # t - r

            alpha = np.arctan2(rel_pos[1], rel_pos[0]) #alpha is the angle between x-axis and the line between the drones

            #alpha = np.arctan(rel_pos[1]/rel_pos[0]) # the angle between x-axis and the line bweteen the drones
            #testar en annan arctan func
            #if isinstance(reciever, Base): # for when bases check for detection
            #    phi_r = 0
            #else:
            #    phi_r = alpha - reciever.state.theta 
            #    phi_r = np.arctan2(np.sin(phi_r), np.cos(phi_r)) # normalize phi to between [-pi,pi]

            phi_r = 0 # all entities see in all directions uniformly
            #phi_t = 0

            if isinstance(transmitter, Base):
               phi_t = 0 ## bases and emitter send in all directions with the same power 
            else:
               phi_t = alpha - transmitter.state.theta + np.pi
                #phi_t = alpha - transmitter.state.theta 
               phi_t = np.arctan2(np.sin(phi_t), np.cos(phi_t)) # "normalizing the phi_t"

            if abs(phi_r) > np.pi/2 or abs(phi_t) > np.pi/2: # the drones do not look at each other
                SNR = 0
            else:
                SNR = transmitter.transmit_power * np.cos(phi_t) * np.cos(phi_r) / (
                    np.linalg.norm(rel_pos) * reciever.internal_noise)**2

            #print(f"reciever {reciever.name}, transmitter {transmitter.name}, phi_r/phi_t: {phi_r}/{phi_t} SNR: ", SNR)

            for jammer in jammers:
                #Start jamming
                rel_pos_j = reciever.state.p_pos - jammer.state.p_pos
                SNR = SNR / (1 + 3 * ((np.linalg.norm(rel_pos_j))**(-2)))

            return SNR
        
        else:
            # SNR calculation in the isotropic sending and receiving scenarios 
            rel_pos = transmitter.state.p_pos - reciever.state.p_pos
            
            transmitted_power = transmitter.transmit_power/(np.linalg.norm(rel_pos) * reciever.internal_noise)**2

            for jammer in jammers:
                #Start jamming
                rel_pos_j = reciever.state.p_pos - jammer.state.p_pos
                transmitted_power = transmitted_power / (1 + 3 * ((np.linalg.norm(rel_pos_j))**(-2)))

            return transmitted_power
        
    
    def full_relative_observation(self, agent):
        # Observing relative positions of all bases and agents (and the enemy)
        base_pos = [base.state.p_pos - agent.state.p_pos for base in self.world.bases]
        other_pos = [other.state.p_pos - agent.state.p_pos for other in self.world.agents if other is not agent]
        if self.observe_self:
            physical_observation = np.concatenate([agent.state.p_pos] + base_pos + other_pos)
        else:
            physical_observation = np.concatenate(base_pos + other_pos)

        if self.antenna_used: # TODO - ändra till relativa antennorienteringar för andra agenter
            own_antenna_direction = agent.state.theta
            antenna_directions = [other.state.theta for other in self.world.agents if other is not agent]
            all_antenna_directions = [own_antenna_direction] + antenna_directions
            physical_observation = np.concatenate([physical_observation, all_antenna_directions])


        communication_observation = []
        communication_observation.append(agent.message_buffer)  # its own observation is seperate - so the policy singels out the correct input corresponding to itself
    
        for other in self.world.agents:  
            if other is not agent: 
                communication_observation.append(other.message_buffer)

    
        # Return one flat observation array
        return np.concatenate([physical_observation, communication_observation])


    def partial_observation(self, agent):
        """
        Here the partial observation model will be implemenetd later on
        """
        pass

    def simple_observation(self, agent):
        """Observation in simple games without communication"""

        base_pos = []
        for base in self.world.bases:
            base_pos.append(base.state.p_pos - agent.state.p_pos)

        other_pos = []
        for other in self.world.agents:
            if other is agent:
                continue
            else:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                #other_vel.append(other.state.p_vel)
                

        ## OBS really need to dubbelcheck if this is the most efficient way to store observations
        # for example: agent's own state can be stored in the same array as all others? 
        return np.concatenate(
            #[agent.state.p_vel]
            [agent.state.p_pos]
            + base_pos
            #+ other_vel
            + other_pos
        )

    
    def observe(self, agent): # these could be included in the Scenario class together with world make/reset
        return self.full_relative_observation(agent)
        #return self.simple_observation(agent)

    #
    def observe_all(self):
        """Return observations for all agents as a dictionary"""
        observations = {agent.name: self.observe(agent) for agent in self.world.agents}
        return observations

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    # OBS maybe add @functools.lru_cache(maxsize=None) - could reduce compute time
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.clock = pygame.time.Clock()
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        self.draw()
        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return

    def draw(self): 
        # clear screen
        self.screen.fill((255, 255, 255))

        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]
        cam_range = max(np.max(np.abs(np.array(all_poses))),1)
        #cam_range = 1 # obs just to check how it looks without scaling

        # update geometry and text positions
        text_line = 0
        for e, entity in enumerate(self.world.entities):
            # geometry
            x, y = entity.state.p_pos
            y *= (
                -1
            )  # this makes the display mimic the old pyglet setup (ie. flips image)
            x = (
                (x / cam_range) * self.width // 2 * 0.9
            )  # the .9 is just to keep entities from appearing "too" out-of-bounds
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2

            # If entity is transmitting, reset intensity to 1
            if entity.state.c >= 1:
                entity.color_intensity = 1.0  # Reset to full intensity

            # Fade transmission intensity over 3 timesteps
            entity.color_intensity = max(entity.color_intensity - 1/3, 0)

            # Adjust color based on intensity
            base_color = entity.color * 200
            fade_factor = entity.color_intensity  # 1.0 when transmitting, fades to 0
            brightened_color = np.minimum(base_color + fade_factor * 150, 255)

            pygame.draw.circle(
                self.screen, brightened_color, (x, y), entity.size * 350 #old color:  entity.color * 200
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
            )  # borders
            scaled_transmission_radius_bases = (self.transmission_radius_bases / cam_range) * (self.width // 2) * 0.9
            if isinstance(entity, Base): # transmit radius for bases
                pygame.draw.circle(
                    self.screen, (0, 0, 0), (x, y), scaled_transmission_radius_bases, 1
                )  # signal transmit radius

            if isinstance(entity, Drone) and self.com_used and not self.antenna_used:
                scaled_transmission_radius_drones = (self.transmission_radius_drones / cam_range) * (self.width // 2) * 0.9
                pygame.draw.circle(
                    self.screen, (0, 0, 0), (x, y), scaled_transmission_radius_drones, 1
                )  # signal transmit radius
            assert (
                0 < x < self.width and 0 < y < self.height
            ), f"Coordinates {(x, y)} are out of bounds."

            # Display entity name next to it
            name_x_pos = x + 10  # Offset slightly to the right
            name_y_pos = y + 10   # Offset slightly below
            self.game_font.render_to(
                self.screen, (name_x_pos, name_y_pos), entity.name, (0, 0, 0), size = 10
            )

            # text - and direction of antenna
            if isinstance(entity, Drone):
                # only draw the arrow representing the antenna if the antenna is controlled by the drone
                if self.antenna_used:
                    # drawing the arrow - direction of antenna
                    arrow_length = 25  # length of the arrow
                    theta = entity.state.theta
                    end_x = x + arrow_length * np.cos(theta)
                    end_y = y - arrow_length * np.sin(theta)  # subtract because of flipped y-axis

                    pygame.draw.line(self.screen, (0, 0, 0), (x, y), (end_x, end_y), 2)  # arrow shaft

                    # the arrowhead is drawn as an polygon
                    head_length = 7  # length of the arrowhead
                    head_angle = np.pi / 6  # angle of the arrowhead
                    left_x = end_x - head_length * np.cos(theta - head_angle)
                    left_y = end_y + head_length * np.sin(theta - head_angle)  # flipped y-axis
                    right_x = end_x - head_length * np.cos(theta + head_angle)
                    right_y = end_y + head_length * np.sin(theta + head_angle)  # flipped y-axis
                    pygame.draw.polygon(
                    self.screen, (0, 0, 0), [(end_x, end_y), (left_x, left_y), (right_x, right_y)])

                if entity.silent: # kanske rita tysta drönare annorlunda? eller markera sändning på ett visst visuellt sätt
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif entity.state.c == None:
                    word = "___"
                else: 
                    word = "msg" 
                #else:
                #    word = (
                #        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                #    )
                #else:
                    #word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.game_font.render_to(
                    self.screen, (message_x_pos, message_y_pos), message, (0, 0, 0)
                )
                text_line += 1

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None