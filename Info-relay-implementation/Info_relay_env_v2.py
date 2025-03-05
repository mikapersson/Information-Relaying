import os

from pettingzoo import ParallelEnv

import gymnasium

from gymnasium import spaces # spaces for action/observation

import numpy as np
from copy import copy, deepcopy
import random

from Info_relay_classes import Drone, Base, Emitter, World, Message

from gymnasium.utils import seeding

# to render basic graphics
import pygame


# class of agents (where drones and emitters(bases are included)) - boolean that shows dynamics or not - future we can have ground/air as well
# vectorized stepping function 

#maybe store all agents (drones + bases + emitters)

#fullständig info, fixed antal agenter i början - coop spel

 
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

##ide: ska det vara egna agent / emitter classer, eller ska allt ingå i env:et?

class Info_relay_env(ParallelEnv):
    metadata = {
        "name": "Info_relay_v2",
        "render_fps": 1
    }

    def __init__(self, num_agents = 10, num_bases = 2, num_emitters = 0, world_size = 1,
                 a_max = 100.0/2, omega_max = 10.0, step_size = 0.05, max_iter = 25, render_mode = None):
        
        self.render_mode = render_mode
        pygame.init()
        self.viewer = None
        self.width = 700#700
        self.height = 700#700
        self.screen = pygame.Surface([self.width, self.height])
        self.max_size = 1
        #self.game_font = pygame.freetype.Font(
        #    os.path.join(os.path.dirname(__file__), "secrcode.ttf"), 24)
        #self.game_font = pygame.font.Font(None, 24)
        self.game_font = pygame.freetype.SysFont('Arial', 24)

        # Set up the drawing window

        self.renderOn = False

        self.n_agents = num_agents
        self.num_bases = num_bases
        self.num_emitters = num_emitters
        self.a_max = a_max
        self.omega_max = omega_max
        self.omega_disc = 36 #the number of dicrete values for omega

        self.SNR_threshold = 1 # threshold for signal detection

        self.world_size = world_size # the world will be created as square. - maybe not used now

        self.h = step_size
        self.max_iter = max_iter # maximum amount of iterations before the world truncates 

        self._seed() # maybe set up seed in another way?? now imprting from gymnasium

        # set all World variables first - contains all enteties
        self.world = self.make_world(num_agents, num_bases, num_emitters)


        self.continous_actions = True # indicies if continous or discrete actions are used
        self.one_hot_vector = False # if continous - this shows if one hot vector or single value representation of actions (output from NN) are used
        

        #self.agents = [agent.name for agent in self.world.agents]
        self.possible_agents = [agent.name for agent in self.world.agents] # self.agents[:]
        
        #all agents get their own obs/action spaces - stored in a Dict structure 
        #obs bases should observe for masseges aswell, right? They can use the same obs_space with different masking
        dim_p = self.world.dim_p
        """
        self.observation_spaces = spaces.Dict(
            {agent: spaces.Box(low = -np.inf, high = np.inf, 
            shape = (2+dim_p*self.n_agents + dim_p*self.num_bases + dim_p*self.num_emitters,), dtype = np.float64)  
            for agent in self.possible_agents} ## kanske BOX(num total agents+other)?
            #shape: the positions of all enteties in the world as well as the velocity of itself
        ) #OBS!! fixa till och dubbelkolla - behöver även kunna observera meddelanden? Hur kodas?
        """"""
        self.observation_spaces = {agent.name: spaces.Dict({
                "physical_observation": spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape = (dim_p*(self.n_agents-1) + dim_p*self.num_bases + dim_p*self.num_emitters,), 
                    dtype=np.float64
                ),
                "communication_observation": spaces.MultiDiscrete( # currently not observing history
                    [100] * (self.n_agents * agent.message_buffer_size * 3) ) # 100 is an arbitrary upper limit of the values 
            }) for agent in self.world.agents}
        # OBS could maybe use Box() for com obs?? Could be weird if they only observe discrete values?
        
        """
        self.observation_spaces = {
            agent.name: spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(dim_p * (self.n_agents - 1) + dim_p * self.num_bases + dim_p * self.num_emitters + self.n_agents * agent.message_buffer_size * 3,),
                dtype=np.float64
            ) for agent in self.world.agents
        }
        
        """
        self.observation_spaces = spaces.Dict(
            {agent: spaces.Box(low = -np.inf, high = np.inf, 
            shape = (dim_p*self.n_agents-2 + dim_p*self.num_bases + dim_p*self.num_emitters,), dtype = np.float64)  
            for agent in self.possible_agents} ## kanske BOX(num total agents+other)?
            #shape: the positions of all enteties in the world as well as the velocity of itself
        )
        """
        """
        self.action_spaces = spaces.Dict({agent: spaces.Box(
            low = np.array([-self.a_max, -self.a_max, -self.omega_max]), 
            high = np.array([self.a_max, self.a_max, self.omega_max]), dtype=np.float64) 
            for agent in self.possible_agents}) # here we have three actions, acceleration in x/y and rotation
        # also need descrete actions for communication!
        """
        """
        # just for testung the simple version when training
        self.action_spaces = spaces.Dict({agent: spaces.Box(
            low = np.array([-self.a_max, -self.a_max]), 
            high = np.array([self.a_max, self.a_max]), dtype=np.float64) 
            for agent in self.possible_agents})
        """
        """
        # trying out actions spaces with communications as a Dict structure
        self.action_spaces = {agent.name: spaces.Dict({
                "physical_action": spaces.Box(
                    low=np.array([-self.a_max, -self.a_max, -self.omega_max]),
                    high=np.array([self.a_max, self.a_max, self.omega_max]),
                    dtype=np.float64),
                "communication_action": spaces.MultiDiscrete([agent.message_buffer_size + 1, agent.message_buffer_size + 1]) # c and d actions
            }) for agent in self.world.agents}
        """
        """
        # testing out a entierly discrete action space
        self.action_spaces = {agent.name: spaces.MultiDiscrete(
                            [3,3,self.omega_disc,agent.message_buffer_size+1, agent.message_buffer_size+1]
                            )for agent in self.world.agents}
        """
        # sets the action spaces for the two different continous action space cases
        if self.continous_actions:
            if self.one_hot_vector:
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
                    low=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),  
                    high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),  
                    dtype=np.float64
                ) 
                for agent in self.world.agents
                }


        #self.terminated_agents = set() # keeps track of lost agents
        ## need a way to add new agents during simulation aswell!

        self.transmission_radius = self.calculate_transmission_radius(self.world.bases[0])

        self.recived_messages = [] # an attribute that keeps track of all messages recieved by bases THIS timestep

        


    # these "world" functions could be included in their own class, Scenario, like in MPE -
    # - then pass Info_relay_env variables. Can be changed later on if needed/makes the program easyer  
    def make_world(self, num_agents, num_bases, num_emitters):
        """
        creates a World object that contains all enteties (drones, bases, emitters) in lists
        """
        world = World()

        world.dt = self.h ## step leangth is transferred to the World

        world.dim_c = 2 # communication dimension, not exactly sure how we will use it yet

        world.agents = [Drone() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.size = 0.025 #quite small
            agent.max_speed = 1.0 # can be cnahged later! or different for differnt agents
            agent.u_range = [self.a_max, self.omega_max] # maximum control range = a_max, omega_max
            agent.internal_noise = 1 ## set internal noise level
            agent.color = np.array([0, 0.33, 0.67])

            #agent.transmit_power = 1 # set to reasonable levels

        world.bases = [Base() for _ in range(num_bases)]
        for i, base in enumerate(world.bases):
            base.name = f"base_{i}"
            base.size = 0.050 # the biggest pieces in the world
            base.color = np.array([0.35, 0.85, 0.83])

            #base.transmit_power = 1 # set to reasonable levels

        world.emitters = []
        #world.emitters = [Emitter() for _ in range(num_emitters)]
        for i, emitter in enumerate(world.emitters):
            emitter.name = f"emitter_{i}"
            emitter.color = np.array([0.35, 0.85, 0.35])

            #emitter.transmit_power = 1 # set to reasonable levels

        return world
    
    ## obs could maybe run make_world here also, otherwise potential problem with truncation ([] agent lists)
    def reset_world(self, world, np_random): # np_ranodm should be some sort of seed for reproducability
        """
        Resets the world and all enteties - called inside the reset function.
        Randomly distributes the bases and emitters - all drones start at one of the bases (first in list)
        """
        world.message_ID_counter = 0 # resets the message_ID counter
        for i, base in enumerate(world.bases):
            base.state.p_pos = np_random.uniform(-self.world_size, self.world_size, world.dim_p)
            base.state.p_vel = np.zeros(world.dim_p)
            # set up messages to be send by the bases
            base.message_to_send = [1,0,0,0,[base.name]]
            self.generate_new_base_message(base) # fixes base.message_to_send
        
        world.bases[0].state.p_pos = np_random.uniform(-self.world_size, self.world_size, world.dim_p)
        world.bases[1].state.p_pos = world.bases[0].state.p_pos + np.array([0.99,0])*self.transmission_radius

        for i, emitter in enumerate(world.emitters):
            #emitter.state.p_pos = np_random.uniform(-self.world_size, self.world_size, world.dim_p)
            emitter.state.p_pos = np.zeros(world.dim_p)
            emitter.state.p_vel = np.zeros(world.dim_p)

        for i, agent in enumerate(world.agents):
            #agent.state.p_pos = np.array(world.bases[0].state.p_pos) # see if this works - all should start at the first base
            agent.state.p_pos = np_random.uniform(-self.world_size, self.world_size, world.dim_p) # randomly assign starting location
            agent.state.p_vel = np.zeros(world.dim_p) # now 2d
            agent.state.theta = 0.0 # should this just be one number? or still nparray?
            # initiate the message_buffer so that it always has the same size
            agent.message_buffer = []
            for i in range(agent.message_buffer_size):
                agent.message_buffer.append([0,0,0,0,None]) 
                #agent.message_buffer.append(Message()) # create new message for each slot in buffer
        
        world.agents[0].state.p_pos = world.bases[0].state.p_pos + 0.99*self.transmission_radius/np.sqrt(2)*np.array([1,1])
        world.agents[1].state.p_pos = world.bases[0].state.p_pos + 0.99*self.transmission_radius*np.array([0,1])

    def reset(self, seed=None, options=None): # options is dictionary
        
        #if options is not None:
        #    self.render_mode = options["render_mode"]
        
        if seed is not None:
            self._seed(seed=seed)
        
        self.reset_world(self.world, self.np_random)

        #if seed is not None: 
        #   np.random.seed(seed) # seed for reproducability
        
        # always start at timestep 0
        self.timestep = 0 

        self.agents = copy(self.possible_agents)

        # All this might only be useful (needed) to AEC envs - might use here anyway
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
        observations = self.observe_all()

        return observations, self.infos # while testing
    

    #sets the action in the case where all actions are discrete
    def set_discrete_action(self, action, agent):
        agent.action.u = np.zeros(3)
        agent.action.c = 0
        agent.action.d = 0

        agent.action.u[0] = action[0]*agent.max_speed
        if action[0] == 2:
            agent.action.u[0] = -agent.max_speed
        agent.action.u[1] = action[1]*agent.max_speed
        if action[1] == 2:
            agent.action.u[1] = -agent.max_speed
        agent.action.u[2] = -np.pi + (action[2] / (self.omega_disc-1)) * (2 * np.pi) # sets the omega between -pi and pi 

        agent.action.c = action[3]
        agent.action.d = action[4]


    def set_com_action_one_hot_vector(self, action, agent):
        buffer_size = agent.message_buffer_size + 1

        agent.action.c = np.argmax(action[3 : 3 + buffer_size])  
        agent.action.d = np.argmax(action[3 + buffer_size : 3 + 2 * buffer_size])

    def set_com_action_one_output(self, action, agent):
        agent.action.c = round(action[3] * agent.message_buffer_size) # translates to integer
        agent.action.d = round(action[4] * agent.message_buffer_size)

    # sets action when all outputs are continous
    def set_action(self, action, agent):
        agent.action.u = np.zeros(3)
        agent.action.c = 0
        agent.action.d = 0

        # 2x-1 transforms the input of [0,1] to [-1,1]
        agent.action.u[0] = (2*action[0]-1)*self.a_max # x velocity (or acc)
        agent.action.u[1] = (2*action[1]-1)*self.a_max # y velocity (or acc)
        agent.action.u[2] = (2*action[2]-1)*self.omega_max # omega, controls theta

        # set com action
        if self.one_hot_vector:
            self.set_com_action_one_hot_vector(action, agent)
        else:
            self.set_com_action_one_output(action, agent)
        
   
    """
    #sets the actions for each agent
    def set_action(self, action, agent):
        agent.action.u = np.zeros(3) 
        agent.action.c = 0 # communication of message in buffer
        agent.action.d = 0 # deletion of message in buffer
                                    
        if agent.movable: ## all our agents will be movable,
            agent.action.u[0] = action['physical_action'][0] # x velocity (or acc)
            agent.action.u[1] = action['physical_action'][1] # y velocity (or acc)
        agent.action.u[2] = action['physical_action'][2] # omega, controls theta

        ##set communication action(s)
        agent.action.c = action['communication_action'][0] # OBS dubblecheck 
        agent.action.d = action['communication_action'][1] # OBS dubblecheck 
    """

    def step(self, actions):
       
        rewards = {}    
        
        for agent in self.world.agents: 
            action = actions.get(agent.name, np.zeros(5)) # get the action for each agent - change 3 to num actions in the end
            #print(action)
            if self.continous_actions:
                self.set_action(action, agent)
            else:
                self.set_discrete_action(action, agent)
        """
        for agent in self.world.agents:
            action = actions.get(agent.name, {
                "physical_action": np.zeros(2),  # Default: no movement
                "communication_action": np.array([0, 0]) #  Default: no communication
            })
            print(action)
            self.set_daction(action, agent)
        """

        #print("step", self.timestep)
        # OBS could make world.step() return the control penalties to be added to the reward later!
        # could also just calculate the penalties here if they come only from the actions!
        self.world.step() # the world controls the motion of the agents - (also controls communication ?)
        #print("world step done")

        # run all comunications in the env
        self.communication_kernel()
        #print("communication kernel done")
        #print(self.world.bases[0].message_to_send)
        #print(self.world.bases[1].message_to_send)

        ## here we can look at the rewards - after world step - could be done after observations instead!
        global_reward = self.global_reward()
        for agent in self.world.agents:
            rewards[agent.name] = float(self.reward(agent, global_reward))
            #print(rewards[agent.name])

        # handle termination - termination if agent is killed, 
        # termination for bases when x num masseges has been succesfully delivered -> terminates all
        #terminations could be stored in a self.terminations that updates each round if agnets died
        #there culd be an internal self.bases_termination that declares game end if enough masseges has gone trough
        terminations = {agent.name: False for agent in self.world.agents}

        # handle truncation - save as dict of dicts?
        #OBS - maybe truncations (and terminations/rewards) HAS to be one dict to fit with pettingzoo?
        truncations = {entity.name: False for entity in self.world.entities}
        if self.timestep > self.max_iter - 2:
            #rewards = {agent.name: 0.0 for agent in self.world.agents} # maybe add reward for bases/emitters later on? far future 
            truncations = {entity.name: True for entity in self.world.entities}
            self.agents = []
            self.bases = []
            self.emitters = []

        self.timestep += 1

        # generate new observations
        #observations = {agent.name: None for agent in self.world.agents}
        observations = self.observe_all()
        #print(observations)

        #self.full_absolute_observation(self.world.agents[0])

        # render if wanted
        if self.render_mode == "human":
            self.render()

        infos = {agent.name: {} for agent in self.world.agents}

        #return observations, rewards, terminations, truncations, infos
        return observations, rewards, terminations, truncations, infos
    
    def calculate_action_penalties(self, agent):
        """
        Calculate the penalties each drone recieves for actions. Movement and communication.
        """
        penalties = 0
        penalties += abs(agent.action.u[0]**2*agent.movement_cost)
        penalties += abs(agent.action.u[1]**2*agent.movement_cost)
        penalties += abs(agent.action.u[2]**2*agent.radar_cost)

        # Communication costs 
        if not agent.action.c == 0: # sending
            penalties += abs(agent.transmission_cost)
        # cost to delete?

        # some penalites to help make the continous com actions more robust
        if self.continous_actions:
            if self.one_hot_vector:
                pass
            else:
                pass

        return penalties
    
    def global_reward(self):
        """
        Rewards given to all agents. Given by correctly delivering messages.
        """
        ## need to look at all messages that were delivered THIS timestep
        # could be done with an env attribute that updates each timestep!
        reward = 0
        for msg in self.recived_messages:
            if msg[2] == int(msg[4][-1].split("_")[1]): # correct destination
                reward += 100 # fine tune so that the reward is large enough!

        return reward
                
    
    def reward(self, agent, global_reward): ## OBS this could be put into the Scenario class
        """ 
        The reward given to each agent - could be made up of multiple different rewards in different functions
        """
        #dist = 0
        #for base in self.world.bases:
            #dist2 += np.sum(np.square(agent.state.p_pos - base.state.p_pos)) # want to minimize distance to bases
        #    dist += np.linalg.norm(agent.state.p_pos - base.state.p_pos)
        #for other in self.world.agents:
        #    dist2 -= np.sum(np.square(agent.state.p_pos - other.state.p_pos)) # want to maximize distance to other agents

        return -self.calculate_action_penalties(agent) + global_reward


    ## some version of this could be used in the beginning so that agents dont run away by mistake?
    def bound(x):
            if x < 0.9:
                return 0
            if x < 1.0:
                return (x - 0.9) * 10
            return min(np.exp(2 * x - 2), 10)
    
    def one_hot_index(vector):
        """
        Helpfunction to get the index corresponding to the one-hot-element
        Vector: list type object of ints (0s and one 1)
        """
        return vector.index(1)
    
    def get_entity_by_name(self, name):
        """
        Returns the base instance correpsonding to a name
        """
        for entity in self.world.agents + self.world.bases:
            if entity.name == name:
                return entity
        
        return None ## The entity does not exist
    

    def check_message_buffer(self, agent, number):
        """Returns a list of the indices in the message buffer that correspond to the given number.
        0 for empty and 1 for non-empty"""
        indices = []
        for index, message in enumerate(agent.message_buffer):
            if message[0] == number: # the first index of each message shows if it exists or not
                indices.append(index) 
        return indices
    
    
    def generate_new_base_message(self, base):
        """
        Generates a new message for a particular base
        """
        base.message_to_send[1] = self.world.message_ID_counter # update message ID
        self.world.message_ID_counter += 1 # update the global message ID counter

        # randomly choosing the destination - excludes itself from the list of possible choices
        possible_destinations = [i for i in range(self.num_bases) if i != base.get_index()] # OBS if num_bases will change during simulation - double check
        base.message_to_send[2] = int(np.random.choice(possible_destinations)) 

    """
    def generate_new_base_message(self, base):
        """"""
        Generates a new message for a particular base - using the Message class
        """"""
        base.message_to_send.exist = 1
        base.message_to_send.ID = self.world.message_ID_counter # update message ID
        self.world.message_ID_counter += 1 # update the global message ID counter

        # randomly choosing the destination - excludes itself from the list of possible choices
        possible_destinations = [i for i in range(self.num_bases) if i != base.get_index()] # OBS if num_bases will change during simulation - double check
        base.message_to_send.destination = int(np.random.choice(possible_destinations)) 

        base.message_to_send.history = base
    """

    def base_com(self):
        """
        Returns all messages from all bases.
        Called from the communication kernel
        """
        messages_bases = []
        
        for base in self.world.bases:
           # toss a coin to send message, the prob of 0.5 can be changed
            if np.random.binomial(1, 0.5) == 1: 
                messages_bases.append(deepcopy(base.message_to_send)) # add copy, avoid referece errors later
                #messages_bases.append(base.message_to_send) # no need for a copy here?
                base.state.c = 1
            else:
                base.state.c = 0 # no message sent 

            #update to new message:
            if np.random.binomial(1, 0.1) == 1:
                # update ID - world variable to keep track of all IDs
                self.generate_new_base_message(base)
            else:
                pass # no update of message 

        return messages_bases 
    
    ## OBS need to be rewritten with one-hot-vector or not
    def delete_message_from_buffer(self, agent):
        """
        Checks if a message is to be deleted and removes it by setting all to 0
        """
        if agent.action.d == 0: ## OBS need to convert one-hot-vector to index!
            pass # no message to be deleted
        elif agent.action.d-1 in self.check_message_buffer(agent, 0): # OBS!! only for testing - passes if trying to delete non existing message
            pass
        else:
            #agent.message_buffer[agent.action.d - 1] = [0,0,0,0,[]]
            # more memory efficient way of deleting messages - not a new list: 
            agent.message_buffer[agent.action.d - 1].clear()  # Removes elements in-place
            agent.message_buffer[agent.action.d - 1].extend([0,0,0,0,[]])  # Avoids reallocating memory

    
    def update_message_time(self, agent):
        """
        Updates the time tracker for each message in each agent's message buffer: +1 per time step
        """
        message_indices = self.check_message_buffer(agent, 1) 
        for index in message_indices:
            agent.message_buffer[index][3] += 1

    
    def communication_kernel(self):
        """
        Handels all communications in the env. An implementation of the communication kernel from the pdf
        Possible to run this from the observe function. 
        Messages are observed? - or only passivly passed to the agents?
        Here: passively passed to the agents by the env - i think reasonable
        Differs slighly from the description in the pdf but not much
        """

        #OBS: message buffer: each row is a message, contains five attributes
        # (0,0,0,0,0) == (exist(int),ID(int),dest(int),num_steps(int),history(list[Entities]))

        messages_bases = self.base_com() # M_ba
        messages_agents = [] # M_ag
        for agent in self.world.agents:
            agent.state.c = 0
            if agent.action.c == 0:
                pass # nothing is transmitted 
            elif agent.action.c-1 in self.check_message_buffer(agent, 0): # the transmitted message does not exist
                pass # OBS: fix better implementation - only to check functionality
            else:
                # add the message from the buffer to the transmit list that correspond to the action
                messages_agents.append(deepcopy(agent.message_buffer[agent.action.c - 1])) 
                #messages_agents.append(agent.message_buffer[agent.action.c - 1][:]) # shallow copy - works with history being only the latest entity
                #essages_agents.append([x[:] if isinstance(x, list) else x for x in agent.message_buffer[agent.action.c - 1]])
                #print(f'{agent} message: {agent.message_buffer[agent.action.c - 1]}')

                agent.state.c = 1


            # deletes message from message buffer
            self.delete_message_from_buffer(agent)
        #print(messages_agents)
        messages = messages_agents + messages_bases
        #print("Messages:", messages)
        for agent in self.world.agents: # step 3 from the com kernel in the pdf
            #print("\n", agent.name)
            pre_buffer = []
            # check each transmitted messaged if it reached this agent and add them to the pre-buffer
            for _, message in enumerate(messages):
                transmitter = self.get_entity_by_name(message[4][-1]) # the latest entity to transmit

                # skips message if the agent who is listening sent it
                if transmitter.name == agent.name: # transmitter is agent does not seem to work here
                    continue

                SNR = self.calculate_SNR(agent, transmitter)
                if self.check_signal_detection(SNR):
                    pre_buffer.append(deepcopy(message)) # need to make copy as to not change later by reference - slower execution
                    #pre_buffer.append(message[:]) # shallow copy
                    #pre_buffer.append([x[:] if isinstance(x, list) else x for x in message]) # 'deep'copy - non recursive
                    pre_buffer[-1].append(SNR) # adding SNR to message to sort later - removing when adding to the buffer
                else:
                    pass # no message recieved - should something be done here?

            # sort the pre-buffer in descending SNR 
            pre_buffer.sort(key=lambda messag: messag[-1], reverse=True)
            # could be used to sort: better time complexity:
            #pre_buffer = heapq.nlargest(len(pre_buffer), pre_buffer, key=lambda message: message[-1])

            #pre_buffer = [deepcopy(messages[0])] # testing the code 
            #pre_buffer[0].append(SNR) 
            #print("pre_buffer:", pre_buffer)

            # update the buffer according to the rules from the pdf - only update if there is space in buffer
            available_indices = self.check_message_buffer(agent, 0)

            if len(available_indices) > len(pre_buffer): # there are enough places in the buffer for all messages
                sampled_indices = random.sample(available_indices, len(pre_buffer)) # take out as many indices as there are messages
            else: # there are not enough places in the buffer for all messages
                sampled_indices = available_indices # here we take out as many messages from the pre_buffer as there are slot in the buffert

            for i, index in enumerate(sampled_indices):
                pre_buffer[i].pop() # remove SNR from the message
                pre_buffer[i][4].append(agent.name) # adding the agent to the history
                # adding the message to the buffer - adding copy as not to have weird reference bugs later
                agent.message_buffer[index] = deepcopy(pre_buffer[i])
                #agent.message_buffer[index] = pre_buffer[i][:]
                #agent.message_buffer[index] = [x[:] if isinstance(x,list) else x for x in pre_buffer[i]]
                

            # update the time of all non-empty message_buffer slots
            self.update_message_time(agent)

            #print(f"{agent.name}'s message_buffer: {agent.message_buffer}")


        self.recived_messages.clear() # reset this for each timestep, used to keep track of messages recieved by bases now
        # do the same thing but for bases
        for base in self.world.bases: # step 4 from pdf
            # check each transmitted messaged if it reached this base and then add it to the (inf big) message buffer
            # Only adds messages that has ben sent by agents?? Chnage messages to messages_agents!
            for _, message in enumerate(messages_agents):
                transmitter = self.get_entity_by_name(message[4][-1]) # the latest entity to transmit
    
                if transmitter.name == base.name: # transmitter is base does not seem to work here??
                    continue # skips this iteration if the sent message is from the base itself

                SNR = self.calculate_SNR(base, transmitter)
                if self.check_signal_detection(SNR):
                    pass
                    base.message_buffer.append(deepcopy(message))
                    #base.message_buffer.append(message[:])
                    #base.message_buffer.append([x[:] if isinstance(x, list) else x for x in message])
                    # update history
                    base.message_buffer[-1][4].append(base.name)
                    # update the time here as the time for delivered messages will stop ticking
                    base.message_buffer[-1][3] += 1

                    self.recived_messages.append(message) # OBS no copy <-- should not be changed 
                else:
                    pass # no message recieved - should something be done here?

            #print(f"{base.name}'s message_buffer: {base.message_buffer}")

        # clear memory of the lists
        #messages.clear()
        #messages_agents.clear()
        #messages_bases.clear()


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
    """
    def calculate_SNR(self, agent, other): ## Change names to ransimtter/reciever!
        # agent is reciever, r, other is transmitter, t
        SNR = 0
        rel_pos = other.state.p_pos - agent.state.p_pos # t - r

        alpha = np.arctan2(rel_pos[1], rel_pos[0]) #alpha is the angle between x-axis and the line between the drones
      
        #alpha = np.arctan(rel_pos[1]/rel_pos[0]) # the angle between x-axis and the line bweteen the drones
        #testar en annan arctan func
        if isinstance(agent, Base): # for when bases check for detection
            phi_r = 0
        else:
            phi_r = alpha - agent.state.theta 
            phi_r = np.arctan2(np.sin(phi_r), np.cos(phi_r)) # normalize phi to between [-pi,pi]
        
        if isinstance(other, Base) or isinstance(other, Emitter):
            phi_t = 0 ## bases and emitter send in all directions with the same power 
        else:
            phi_t = alpha - other.state.theta + np.pi
            #phi_t = alpha - other.state.theta 
            phi_t = np.arctan2(np.sin(phi_t), np.cos(phi_t))
        
        if abs(phi_r) > np.pi/2 or abs(phi_t) > np.pi/2: # the drones do not look at each other
            SNR = 0
        else:
            SNR = other.transmit_power * np.cos(phi_t) * np.cos(phi_r) / (
                np.linalg.norm(rel_pos) * agent.internal_noise)**2

        #print(f"reciever {agent.name}, transmitter {other.name}, phi_r/phi_t: {phi_r}/{phi_t} SNR: ", SNR)

        return SNR
    """

    def calculate_SNR(self, reciever, transmitter):
        """
        Calculates the SNR between transmitter and reciever. It is assumed that all entities can listen uniformly in all directions.
        All bases send uniformly in all directions. All agents send in the direction of its antenna
        """
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
        
        if isinstance(transmitter, Base) or isinstance(transmitter, Emitter):
            phi_t = 0 ## bases and emitter send in all directions with the same power 
        else:
            phi_t = alpha - transmitter.state.theta + np.pi
            #phi_t = alpha - transmitter.state.theta 
            phi_t = np.arctan2(np.sin(phi_t), np.cos(phi_t))
        
        if abs(phi_r) > np.pi/2 or abs(phi_t) > np.pi/2: # the drones do not look at each other
            SNR = 0
        else:
            SNR = transmitter.transmit_power * np.cos(phi_t) * np.cos(phi_r) / (
                np.linalg.norm(rel_pos) * reciever.internal_noise)**2

        #print(f"reciever {reciever.name}, transmitter {transmitter.name}, phi_r/phi_t: {phi_r}/{phi_t} SNR: ", SNR)

        return SNR


    def base_observation(self, base): # anropa i reward-funktionen?
        """
        The base check if any signals are detected and if any of them are correctly delivered. 
        Returns true if correct signal detected?

        OBS - not currently used: could be used if sets of transmitted messages are not used
        """

        for other in self.world.bases:
            if other is base:
                continue
            SNR = self.calculate_SNR(base, other)
            if self.check_signal_detection(SNR): # if signal detected
                pass

        for agent in self.world.agents:
            SNR = self.calculate_SNR(base, other)
            if self.check_signal_detection(SNR):
                pass
        
    """
    def full_relative_observation(self, agent):
        
        #Observes everything perfectly in the env, return the relative coordinates of everything
          
        base_comm = []
        base_pos = []
        for base in self.world.bases:
            base_pos.append(base.state.p_pos - agent.state.p_pos)
            SNR = self.calculate_SNR(agent, base)
            signal_detected = self.check_signal_detection(SNR) # OBS måste kolla ifall signal sänds också
            if signal_detected: # add communication - later
                pass
            ## the probability of detecting this signal is prop to distance and noise, 
            # the direction of only the agent's antenna --> cos(\phi_r) = 1 

        other_pos = []
        other_vel = []
        other_comm = []
        for other in self.world.agents:
            if other is agent:
                continue
            else:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                #other_vel.append(other.state.p_vel)

                #communication part
                # first check if signal is detected, then add all detected signals to the observations
                SNR = self.calculate_SNR(agent, other)
                signal_detected = self.check_signal_detection(SNR) # OBS måste kolla ifall signal sänds också
                # now add the signal to comm list
                if signal_detected:
                    pass
                

        ## OBS really need to dubbelcheck if this is the most efficient way to store observations
        # for example: agent's own state can be stored in the same array as all others? 
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + base_pos
            #+ other_vel
            + other_pos
            #+ base_comm
            #+ other_comm
        )
    """
    """
    def full_relative_observation(self, agent):
        
        #Observes everything perfectly in the env, return the relative coordinates of everything
          
        base_comm = []
        base_pos = []
        for base in self.world.bases:
            base_pos.append(base.state.p_pos - agent.state.p_pos)
            SNR = self.calculate_SNR(agent, base)
            signal_detected = self.check_signal_detection(SNR) # OBS måste kolla ifall signal sänds också
            if signal_detected: # add communication - later
                pass
            ## the probability of detecting this signal is prop to distance and noise, 
            # the direction of only the agent's antenna --> cos(\phi_r) = 1 

        other_pos = []
        other_vel = []
        other_comm = []
        for other in self.world.agents:
            if other is agent:
                continue
            else:
                other_pos.append(other.state.p_pos - agent.state.p_pos)
                #other_vel.append(other.state.p_vel)

                #communication part
                # first check if signal is detected, then add all detected signals to the observations
                SNR = self.calculate_SNR(agent, other)
                signal_detected = self.check_signal_detection(SNR) # OBS måste kolla ifall signal sänds också
                # now add the signal to comm list
                if signal_detected:
                    pass
                

        ## OBS really need to dubbelcheck if this is the most efficient way to store observations
        # for example: agent's own state can be stored in the same array as all others? 
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + base_pos
            #+ other_vel
            + other_pos
            #+ base_comm
            #+ other_comm
        )
    """
    
    def full_relative_observation(self, agent):
        # Observing relative positions of all bases, agents, and emitters
        base_pos = [base.state.p_pos - agent.state.p_pos for base in self.world.bases]
        other_pos = [other.state.p_pos - agent.state.p_pos for other in self.world.agents if other is not agent]

        physical_observation = np.concatenate(base_pos + other_pos)

        """
        # Construct communication observation (including self)
        communication_observation = []
        
        #OBS should the agents always observe its own buffer first?
        for observed_agent in self.world.agents:  # Loop over all agents (including self)
            for msg in observed_agent.message_buffer:  # Read messages from each agent's buffer
                msg_id = min(msg[1], 99)  # Clamp to [0, 99]
                destination = min(msg[2], 99)  # Clamp to [0, 99]
                timestamp = min(msg[3], 99)  # Clamp to [0, 99]
                communication_observation.extend([msg_id, destination, timestamp])

        # Pad communication observations to maintain fixed shape
        total_comm_size = self.num_agents * agent.message_buffer_size * 3
        communication_observation.extend([0] * (total_comm_size - len(communication_observation)))

        # Convert to numpy array
        communication_observation = np.array(communication_observation, dtype=np.int64)

        return {
            "physical_observation": physical_observation,
            "communication_observation": communication_observation
        }
        """
        communication_observation = []
    
        for observed_agent in self.world.agents:  # Loop over all agents (including self)
            for msg in observed_agent.message_buffer:  # Read messages from each agent's buffer
                msg_id = msg[1]  # Continuous signal
                destination = msg[2]  
                timestamp = msg[3]
                communication_observation.extend([msg_id, destination, timestamp])
        """
        # Pad communication observations to maintain a fixed shape
        total_comm_size = self.n_agents * agent.message_buffer_size * 3
        if len(communication_observation) < total_comm_size:
            padding = [0.0] * (total_comm_size - len(communication_observation))
            communication_observation.extend(padding)
        """

        # Convert to numpy array
        communication_observation = np.array(communication_observation, dtype=np.float64)
    
        # Return one flat observation array
        return np.concatenate([physical_observation, communication_observation])

    def partial_observation(self, agent):
        """
        Here the partial observation model will be implemenetd later on
        """
        pass
    
    def observe(self, agent): # these could be included in the Scenario class together with world make/reset
        return self.full_relative_observation(agent)

    def observe_all(self):
        """Return observations for all agents as a dictionary"""
        return {agent.name: self.observe(agent) for agent in self.world.agents}


    #def render(self):
    #    pass # try to import something already done?
    #        # or maybe use basic graphics in python for simplicity - later unity 

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
        cam_range = np.max(np.abs(np.array(all_poses)))
        cam_range = 1 # obs just to check how it looks without scaling

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
            if entity.state.c == 1:
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
            scaled_transmission_radius = (self.transmission_radius / cam_range) * (self.width // 2) * 0.9
            if isinstance(entity, Base): # transmit radius for bases
                pygame.draw.circle(
                    self.screen, (0, 0, 0), (x, y), scaled_transmission_radius, 1
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
                    word = "communication state not defined"
                else: 
                    word = " "
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