from pettingzoo import ParallelEnv

import gymnasium

from gymnasium import spaces # spaces for action/observation

import numpy as np
from copy import copy, deepcopy
import random
import os
from pathlib import Path
import csv

# to be able to import from current directory in both linux and windows
try:
    from Info_relay_classes import Drone, Base, Emitter, World, EvaluationLogger#, Message  
except ImportError:
    from .Info_relay_classes import Drone, Base, Emitter, World, EvaluationLogger#, Message

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
        "render_fps": 5
    }

    def __init__(self, num_agents = 1, num_bases = 2, num_emitters = 0, world_size = 1,
                 a_max = 0.1, omega_max = np.pi/4, step_size = 1, max_cycles = 25, 
                 continuous_actions = True, one_hot_vector = False, antenna_used = True, 
                 com_used = True, num_messages = 1, base_always_transmitting = True, 
                 observe_self = True, render_mode = None,
                 pre_determined_scenario = False, num_CL_episodes = 0):
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

        self.num_CL_episodes = num_CL_episodes

        self.h = step_size
        self.max_iter = max_cycles # maximum amount of iterations before the world truncates - OBS renamed the inupt to more closely match benchmarl

        self._seed() # maybe set up seed in another way?? now imprting from gymnasium

        self.num_messages = num_messages # to keep track of the number of messages = to the number of message buffer slots the agents will have

        # set all World variables first - contains all enteties
        self.world = self.make_world(num_agents, num_bases, num_emitters)

        self.continuous_actions = continuous_actions # indicies if continous or discrete actions are used
        self.one_hot_vector = one_hot_vector # if continous - this shows if one hot vector or single value representation of actions (output from NN) are used
        self.pre_determined_scenario = pre_determined_scenario
        if self.pre_determined_scenario:
            self.evaluation_logger = EvaluationLogger()
            self.pre_loaded_scenarios = []
            self.scenario_index_counter = 0
            self.evaluation_logger.update_episode_index(self.scenario_index_counter)
            self.read_scenario_csv()
        else:
            self.evaluation_logger = None
        self.antenna_used = antenna_used #OBS ändra när antennen ska styras igen
        self.com_used = com_used #OBS - communications can be turned of to test just the movement in different tasks
        self.observe_self = observe_self
        self.angle_coord_rotation = 0 # declared as a class variabel to be reach in observation and in setting actions

        self.base_always_transmitting = base_always_transmitting # decides if the base is sending every time step. If false, the base send sporadically

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
        self.world.transmission_radius = self.transmission_radius_bases

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
            agent.max_speed = self.a_max
            agent.u_range = [self.a_max, self.omega_max] # maximum control range = a_max, omega_max
            agent.internal_noise = 1 ## set internal noise level
            agent.color = np.array([0, 0.33, 0.67])
            agent.message_buffer_size = self.num_messages

            #agent.transmit_power = 1 # set to reasonable levels

        world.emitters = [Emitter() for _ in range(num_emitters)]
        for i, emitter in enumerate(world.emitters):
            emitter.name = f"jammer_{i}"
            emitter.size = 0.025/2
            emitter.max_speed = self.a_max
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

    def generate_base_positions(self, R):
        """Generate random base positions with equal spacing.
        Used in reset_world to place bases."""
        
        positions = np.array([[0.0, 0.0], [R, 0.0]])

        return positions   
    
    
    def generate_agent_positions(self, np_random, base_positions, spawn_radius, n_entities):
        """
        Generate positions for agents inside a circular disk with a certain radius. 
        The center of the disk is the the middlepoint of the bases.  
        """
        positions = []
    
        center = np.mean(base_positions, axis=0)  # Midpoint of bases
        for i in range(n_entities):
            radius_agent = np.sqrt(np_random.uniform(0, 1)) * spawn_radius  # Random radius
            angle = np_random.uniform(0, 2 * np.pi)  # Random angle
            offset = np.array([radius_agent * np.cos(angle), radius_agent * np.sin(angle)])  # Convert to Cartesian
            positions.append(center + offset)

        return np.array(positions)
    
    def generate_jammer_positions(self, np_random, base_positions, n_entities, transmission_radius):
        """
        Generate positions for jammers inside a geometric shape consisting of a rectangle with height 3 * Rcom and width R,
        flanked on its right and left sides by semicircles of radius 1.5 * Rcom. The left and right semicircles are centered
        at the positions of the transmitting and receiving bases, respectively.

        Assumes that that the bases lay on the x-axis
        """

        positions = []

        for i in range(n_entities):

            reject_sample = True
            while reject_sample:
                x_pos = np_random.uniform(base_positions[0][0] - 1.5*transmission_radius, base_positions[1][0] + 1.5*transmission_radius)
                y_pos = np_random.uniform(base_positions[0][1] - 1.5*transmission_radius, base_positions[0][1] + 1.5*transmission_radius)

                position = np.array([x_pos, y_pos])

                reject_sample = Emitter.check_boundary(base_positions, position, transmission_radius)

            positions.append(position)

        return positions

    def get_max_base_distance(self, world):
        """ The function return the meximum allowed distance between the bases - depends on if curriculum learning is used or not """
        #if self.curriculum_learning: - gammal CL implementation
            #R_min = self.transmission_radius_bases * self.n_agents
            #pot_rmax = R_min + self.transmission_radius_bases * (self.n_agents + 4) * self.episode_counter/1000
            #R_max = min(R_min, pot_rmax)
        #else:
        R_max = self.transmission_radius_bases * (self.n_agents + 4)
        
        return R_max
    
    def reset_world(self, world, np_random): # np_ranodm should be some sort of seed for reproducability
        """
        Resets the world and all enteties - called inside the reset function.
        Randomly distributes the bases and emitters - all drones start at one of the bases (first in list)
        """
        for i, emitter in enumerate(world.emitters):
            #emitter.state.p_pos = np_random.uniform(-self.world_size, self.world_size, world.dim_p)
            emitter.state.p_pos = np.zeros(world.dim_p)
            emitter.state.p_vel = np.zeros(world.dim_p)


        R_max = self.get_max_base_distance(world)    

        self.R_half = np_random.uniform(self.transmission_radius_bases * self.n_agents, R_max)/2
        self.R = self.R_half * 2
        if self.num_bases == 3:
            self.R_half = self.transmission_radius_bases * (2 + 1)/2 # always the same distance as 2 agents - does not work otherwise

        base_positions = self.generate_base_positions(self.R)
        world.base_positions = base_positions
        #base_positions = self.generate_base_positions(np_random, self.radius * np_random.uniform(0.8, 1.0))
        for i, base in enumerate(world.bases):
            #base.state.p_pos = np_random.uniform(-self.world_size, self.world_size, world.dim_p)
            base.state.p_pos = base_positions[i]
            base.state.p_vel = np.zeros(world.dim_p)

            base.color_intensity = 0 # resets graphics so it does not look like it is sending
            
            base.state.p_pos_history = []

        world.R = np.linalg.norm(world.bases[0].state.p_pos - world.bases[1].state.p_pos)

        world.bases[0].state.c = 1
        world.bases[1].state.c = 0
        
        world.bases[1].silent = True # only one base sending
        world.bases[0].generate_messages = False # the same message all the time
        world.bases[0].silent = False 
        world.bases[1].message_buffer = False # reset the message buffer 
            

        # Compute the midpoint of all bases
        #base_positions = np.array(positions)
        self.center = np.mean(base_positions, axis=0)  # Midpoint of bases

        # I feel lie the center point should be contained in world but idk
        world.center = self.center

        #radius = self.radius * min(self.episode_counter / 2500, 1) # increases from 0 to 1 
        #radius = self.radius*2
        spawn_radius = self.R_half*1.2
        #radius = self.radius

        agent_positions = self.generate_agent_positions(np_random, base_positions, spawn_radius, self.n_agents)

        for i, agent in enumerate(world.agents):
            #agent.state.p_pos = np.array(world.bases[0].state.p_pos) # all starts at the first base
            #agent.state.p_pos = np_random.uniform(-self.world_size*3, self.world_size*3, world.dim_p) # randomly assign starting location in a square
            agent.state.p_pos = agent_positions[i]
            agent.state.p_vel = np.zeros(world.dim_p) 

            agent.state.theta = np.random.uniform(0,2*np.pi)
            # initiate the message_buffer so that it always has the same size
            agent.message_buffer = False 
            agent.state.c = 0 # ingen agent börjar med meddelande - alltså sänder ingen i början - kanske inte behöver denna - kör bara .message_buffer
            agent.state.p_pos_history = [] # reset the position history - for visuals

        emitter_positions = self.generate_jammer_positions(np_random, base_positions, self.num_emitters, self.transmission_radius_bases)

        for i, emitter in enumerate(world.emitters):
            #agent.state.p_pos = np.array(world.bases[0].state.p_pos) # all starts at the first base
            #agent.state.p_pos = np_random.uniform(-self.world_size*3, self.world_size*3, world.dim_p) # randomly assign starting location in a square
            emitter.state.p_pos = emitter_positions[i]
            emitter.state.p_vel = np.zeros(world.dim_p) 
            emitter.generate_action(self.R)
            emitter.state.theta = 0.0 # TODO - maybe randomize?
            emitter.state.p_pos_history = []

    def apply_pre_loaded_scenario(self):
        """
            Reset scenario based on csv file loaded via read_scenario_csv()
        """
        print("loading scenario with id:", self.pre_loaded_scenarios[self.scenario_index_counter][0])
        scenario = self.pre_loaded_scenarios[self.scenario_index_counter]
        scenario = [float(entry) for entry in scenario]
        self.R = scenario[1] # distance between bases
        # R_com = 2 # communication distance (always equal to 1)
        # R_a = 3 # Not sure. Always 0
        # p_tx_x = 4 # This stuff has to do with base positions. Fuck that
        # p_tx_y = 5 # ^
        # p_rx_x = 6 # ^
        # p_rx_y = 7 # ^
        # jammer_x = 8 # Jammer position x
        # jammer_y = 9 # Jammer position y
        # jammer_dx = 10  # Jammer velocity x
        # jammer_dy = 11 # Jammer velocity y
        # agent1_x = 12 #agent position x
        # agent1_y = 13 # agent position y
        # agent1_phi = 14 # agent antenna direction

        base_positions = self.generate_base_positions(self.R)
        self.world.base_positions = base_positions

        for i, base in enumerate(self.world.bases):
            #base.state.p_pos = np_random.uniform(-self.world_size, self.world_size, world.dim_p)
            base.state.p_pos = base_positions[i]
            base.state.p_vel = np.zeros(self.world.dim_p)

            base.color_intensity = 0 # resets graphics so it does not look like it is sending
            
            base.state.p_pos_history = []

        self.world.R = np.linalg.norm(self.world.bases[0].state.p_pos - self.world.bases[1].state.p_pos)

        self.world.bases[0].state.c = 1
        self.world.bases[1].state.c = 0
        
        self.world.bases[1].silent = True # only one base sending
        self.world.bases[0].generate_messages = False # the same message all the time
        self.world.bases[0].silent = False 
        self.world.bases[1].message_buffer = False # reset the message buffer 

        # Compute the midpoint of all bases
        #base_positions = np.array(positions)
        self.center = np.mean(base_positions, axis=0)  # Midpoint of bases

        # I feel lie the center point should be contained in world but idk
        self.world.center = self.center

        for i, agent in enumerate(self.world.agents):
            agent.state.p_pos = np.array([scenario[12 + i * 2], scenario[13 + i * 2]])
            agent.state.p_vel = np.zeros(self.world.dim_p) 

            agent.state.theta = scenario[14 + i * 2]
            # initiate the message_buffer so that it always has the same size
            agent.message_buffer = False 
            agent.state.c = 0 # ingen agent börjar med meddelande - alltså sänder ingen i början - kanske inte behöver denna - kör bara .message_buffer
            agent.state.p_pos_history = [] # reset the position history - for visuals

        for i, emitter in enumerate(self.world.emitters):
            emitter.state.p_pos = np.array([scenario[8], scenario[9]])
            emitter.state.p_vel = np.zeros(self.world.dim_p)
            
            emitter.action.u = np.array([scenario[10], scenario[11]])
            emitter.state.theta = 0.0 # TODO - maybe randomize?
            emitter.state.p_pos_history = []

            # Whack but only do one jammer for now
            break
            

    def read_scenario_csv(self):

        script_dir = os.path.dirname(os.path.abspath(__file__))

        scenario_file = os.path.join(script_dir, f"initial_state_pool/evaluation_states_K{self.n_agents}_n10000.csv")

        with open(scenario_file, 'r') as f:
            csv_reader = csv.reader(f)

            # Header includes description of fields. Unnecessary
            header = next(csv_reader)

            for row in csv_reader:
                self.pre_loaded_scenarios.append(row)


        
    def reset(self, seed=None, options=None): # options is dictionary
        #if options is not None:
        #    self.render_mode = options["render_mode"]

        if seed is not None:
            self._seed(seed=seed)
        
        if self.pre_determined_scenario:
            self.evaluation_logger.write_episode()
            self.apply_pre_loaded_scenario()
            self.scenario_index_counter = self.scenario_index_counter + 1
            self.evaluation_logger.update_episode_index(self.scenario_index_counter)
        else:
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

        self.world.step() # the world controls the motion of the agents

        # run all comunications in the env
        self.communication_kernel()
        
        ## here we can look at the rewards - after world step - could be done after observations instead!
        global_reward = self.global_reward()
        total_action_penalties = 0
        for agent in self.world.agents:
            total_action_penalties += self.calculate_action_penalties(agent)

        for agent in self.world.agents:
            if self.episode_counter < self.num_CL_episodes: # hjälpreward 
                rewards[agent.name] = float(self.reward(agent, global_reward, total_action_penalties)) + agent.reward_bonus
            else: 
                rewards[agent.name] = float(self.reward(agent, global_reward, total_action_penalties))
            agent.reward_bonus = 0
        
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


    def calculate_action_penalties(self, agent, eval_logger = None):
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
                if eval_logger is not None:
                    eval_logger.add_movement(abs(agent.action.u[0]))
            else:
                penalties += abs(agent.action.u[0]**2*agent.movement_cost)
                penalties += abs(agent.action.u[1]**2*agent.movement_cost)
                if eval_logger is not None:
                    eval_logger.add_movement(abs(agent.action.u[0]) + abs(agent.action.u[1]))
            penalties += abs(agent.action.u[2]**2*agent.radar_cost)

        return penalties
    

    def compute_budget(self, K, R, sigma, gamma):
        """
        Compute the budget(w) as described in the prompt.

        Args:
            K (int): Number of agents
            R (float): Distance between transmitter and receiver
            sigma (float): Displacement size
            gamma (float): Discount factor

        Returns:
            float: The computed budget(w)
        """
        Rcom = 1.0

        # Compute T_sharp
        T_sharp = int(np.floor(1.1 * R + 2 * Rcom) / sigma + K)

        # Compute D_k for all agents
        D = np.zeros(K)
        for k in range(1, K+1):
            if k == 1:
                D[k-1] = 1.1 * R + 2 * Rcom
            else:
                D[k-1] = 0.1 * R + (K - k + 1) * Rcom

        # Compute t_start_k and t_stop_k for all agents
        t_start = np.zeros(K, dtype=int)
        t_stop = np.zeros(K, dtype=int)
        for k in range(1, K+1):
            if k == 1:
                t_start[k-1] = 0
                t_stop[k-1] = int(np.ceil(D[0] / sigma))
            else:
                t_start[k-1] = int(np.floor((D[0] - D[k-1]) / sigma)) + (k - 1)
                t_stop[k-1] = T_sharp - (k + 1)

        # Compute budget(w)
        budget = 0.0
        for t in range(T_sharp):
            gamma_t = gamma ** t
            active_agents = sum(
                t_start[k] <= t < t_stop[k] for k in range(K)
            )
            budget += gamma_t * sigma**2 * active_agents
        budget = budget / (gamma ** T_sharp)

        return budget, T_sharp, t_start, t_stop


    def global_reward(self):
        """ Kör Mikas funktion direkt - testar """
        self.discount_factor = 0.99 

        if self.world.bases[1].message_buffer: # meddelandet har levererats (detta tidsteg)
            return self.compute_budget(self.n_agents, self.R, self.a_max, self.discount_factor)[0]
        else:
            return 0

    # def global_reward(self):
    #     """
    #     Rewards given to all agents. Given by correctly delivering messages.
    #     """
    #     reward = 0
    #     self.discount_factor = 0.99 # TODO OBS fixa som input till klassen!! - ska sättas en gång både till envet och experimentet

    #     D_tot = (1 + 0.1*self.n_agents)*self.R + (2 + 0.5*self.n_agents*(self.n_agents - 1))*self.world.transmission_radius

    #     T = (1.1*self.R + 2*self.world.transmission_radius)/self.a_max + self.n_agents

    #     if self.world.bases[1].message_buffer: # meddelandet har levererats (detta tidsteg)
    #         reward = (1 - self.discount_factor**T)/(1 - self.discount_factor) * (1/self.discount_factor**T) * (D_tot/(self.n_agents*T))**2

    #     return reward 

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
        normalized_dist = dist / self.R_half  # Normalize relative to allowed radius

        if normalized_dist < 1.2:
            return 0
        if normalized_dist < 1.3:
            return (normalized_dist - 0.9) * 10
        return min(np.exp(2 * normalized_dist - 2), 10)

    
    def reward(self, agent, global_reward, action_penalties): ## OBS this could be put into the Scenario class
        """ 
        The reward given to each agent - could be made up of multiple different rewards in different functions
        """
        # TODO - update to new reward
        return global_reward - action_penalties #self.calculate_action_penalties(agent)
    

    def get_entity_by_name(self, name):
        """
        Returns the entity instance correpsonding to a name
        """
        for entity in self.world.agents + self.world.bases + self.world.emitters:
            if entity.name == name:
                return entity
        
        return None ## The entity does not exist


    
    def communication_kernel(self, eval_logger = None):
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
                if base.state.c == 1 and not agent.state.c == 1: # if sending
                    SNR = self.calculate_SNR(agent, base, self.world.emitters)
                    if self.check_signal_detection(SNR):
                        agent.message_buffer = True
                        agent.state.c = 1 # not it will send continously
                        agent.reward_bonus += 0.25
                        recieved_message = True
                        if eval_logger is not None:
                            eval_logger.add_air_distance(base, agent)


            if recieved_message: # do not check agents if message recieved from base
                continue

            for other in self.world.agents: # the agents are always transmitting
                if other.name == agent.name:
                    continue
                if other.message_buffer and not agent.message_buffer:
                    SNR = self.calculate_SNR(agent, other, self.world.emitters)
                    if self.check_signal_detection(SNR):
                        agent.message_buffer = True
                        agent.state.c = 1
                        agent.reward_bonus += 0.25
                        other.reward_bonus += 0.25
                        if eval_logger is not None:
                            eval_logger.add_air_distance(other, agent)
                        continue

        for base in self.world.bases: # maybe remove loop - only look at the 2nd base
            for agent in self.world.agents:
                if agent.message_buffer and base.state.c != 1:
                    SNR = self.calculate_SNR(base, agent, self.world.emitters)
                    if self.check_signal_detection(SNR):
                        base.message_buffer = True # the game should end once this condition is met
                        agent.reward_bonus += 1.0
                        if eval_logger is not None:
                            eval_logger.add_air_distance(agent, base)
                        continue
            for other in self.world.bases: # maybe remove loop - only look at the first base
                if other.name == base.name:
                    continue
                if base.state.c == 1:
                    SNR = self.calculate_SNR(base, other, self.world.emitters)
                    if self.check_signal_detection(SNR):
                        base.message_buffer = True
                        eval_logger.add_air_distance(other, base)
                        continue

    
    def check_base_com(self):
        """ Checks if the base should send this timestep """
        if not self.base_always_transmitting:
            #self.world.bases[0].state.c = np.random.binomial(1, 0.2) # singlar slant om den ska sända eller ej
            self.world.bases[0].c = 1 # base is always sending
 

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
            rel_pos = reciever.state.p_pos - transmitter.state.p_pos # t - r

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
               theta = 0 ## bases and emitter send in all directions with the same power 
            else:
               theta = (alpha - transmitter.state.theta + np.pi)%(2*np.pi) - np.pi

            if (theta < - np.pi/2 or np.pi/2 < theta):
                SNR = 0
            else: 
                SNR = SNR = np.linalg.norm([1 + np.cos(np.pi * np.sin(theta)), np.sin(np.pi * np.sin(theta))]) / (np.linalg.norm(rel_pos))**2
                
            for jammer in jammers:
                #Start jamming
                rel_pos_j = reciever.state.p_pos - jammer.state.p_pos
                SNR = SNR / (1 + 3 * ((np.linalg.norm(rel_pos_j))**(-2)))

            return SNR
        
        else:
            # SNR calculation in the isotropic sending and receiving scenarios 
            rel_pos = transmitter.state.p_pos - reciever.state.p_pos
            
            transmitted_power = transmitter.transmit_power/(np.linalg.norm(rel_pos) * reciever.internal_noise)**2

            reciever.current_jamming_factor = 1

            for jammer in jammers:
                #Start jamming
                rel_pos_j = reciever.state.p_pos - jammer.state.p_pos
                reciever.current_jamming_factor /= (1 + 3 * ((np.linalg.norm(rel_pos_j))**(-2)))
                
            transmitted_power = transmitted_power * reciever.current_jamming_factor

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
        #cam_range = 10 # obs just to check how it looks without scaling

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
            x += self.width // 4
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

            entity.state.save_history()
            history_copy = copy(entity.state.p_pos_history)
            for i, pos in enumerate(history_copy):
                # Rescale for screen
                history_copy[i] = ((pos[0] / cam_range) * self.width // 2 * 0.9 + self.width // 4, -(pos[1] / cam_range) * self.height // 2 * 0.9 + self.height // 2)

            if (len(history_copy) > 1):
                pygame.draw.lines(self.screen, brightened_color, False, history_copy, round(entity.size * 350))

            pygame.draw.circle(
                self.screen, brightened_color, (x, y), entity.size * 350 #old color:  entity.color * 200
            )  # 350 is an arbitrary scale factor to get pygame to render similar sizes as pyglet
            pygame.draw.circle(
                self.screen, (0, 0, 0), (x, y), entity.size * 350, 1
            )  # borders
            scaled_transmission_radius_bases = (self.transmission_radius_bases / cam_range) * (self.width // 2) * 0.9
            if isinstance(entity, Base): # transmit radius for bases
                pygame.draw.circle(
                    self.screen, (0, 0, 0), (x, y), scaled_transmission_radius_bases * np.sqrt(entity.current_jamming_factor), 1
                )  # signal transmit radius

            if isinstance(entity, Drone) and self.com_used and not self.antenna_used:
                scaled_transmission_radius_drones = (self.transmission_radius_drones / cam_range) * (self.width // 2) * 0.9
                pygame.draw.circle(
                    self.screen, (0, 0, 0), (x, y), scaled_transmission_radius_drones * np.sqrt(entity.current_jamming_factor), 1
                )  # signal transmit radius
            #assert (
            #    0 < x < self.width and 0 < y < self.height
            #), f"Coordinates {(x, y)} are out of bounds."

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