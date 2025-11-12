from copy import copy
import math
import numpy as np
import csv

from dataclasses import dataclass, field

"""
Taken from pettingzoo MPE and altered 
"""

class EvaluationLogger:
    def __init__(self, directed_transmission = False, K = 0, file = ""):
        self.episode_index = 0
        self.success = np.zeros(10000 + 1, dtype=bool)
        self.R = np.zeros(10000 + 1)
        self.value = np.zeros(10000 + 1)
        self.budget = np.zeros(10000 + 1)
        self.episode_movement = np.zeros(10000 + 1)
        self.episode_air_distance = np.zeros(10000 + 1)
        self.delivery_time = np.zeros(10000 + 1)
        self.directed_transmission = directed_transmission
        self.K = K
        self.file = file.split("/")[-1]
        
        # idx, success, R, value, budget, sum_distance, air_distance, delivery_time, directed_transmission_bool, K, file
        self.f_eval = open('evaluation_log', 'w')
        self.writer_eval = csv.writer(self.f_eval)
        self.writer_eval.writerow(["idx", "sucess", "R", "value", "budget", "sum_distance",
                                   "air_distance", "delivery_time", "directed_transmission", "K", "file"])

    def write_episode(self):
        self.writer_eval.writerow([self.episode_index,
                                   self.success[self.episode_index],
                                   self.R[self.episode_index],
                                   self.value[self.episode_index],
                                   self.budget[self.episode_index],
                                   self.episode_movement[self.episode_index],
                                   self.episode_air_distance[self.episode_index],
                                   self.delivery_time[self.episode_index],
                                   self.directed_transmission,
                                   self.K,
                                   self.file])

    def update_episode_index(self, index):
        self.episode_index = index

    def add_delivery_time(self, time):
        self.delivery_time[self.episode_index] += time

    def set_budget(self, budget):
        self.budget[self.episode_index] = budget

    def set_R(self, R):
        self.R[self.episode_index] = R

    def set_success(self, success):
        self.success[self.episode_index] = success

    def add_movement(self, movement):
        self.episode_movement[self.episode_index] += movement

    def add_air_distance(self, sender, receiver):
        distance = np.linalg.norm(sender.state.p_pos - receiver.state.p_pos)
        self.episode_air_distance[self.episode_index] += distance

    def add_value(self, time_step, reward):
        self.value[self.episode_index] += reward * 0.99 ** time_step

class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None 
        self.p_pos_history = []
        self.p_history_max_length = 200
        # physical velocity
        self.p_vel = None
        # communication utterance
        self.c = None # OBS init all agent with the same (ID)!!!
        # transmitting if True, listening if False. 
        #self.sending = False # obs kalla denna self.c som i envet? eller kanske modda envet?

    def save_history(self):
        self.p_pos_history.append(copy(self.p_pos))
        while (len(self.p_pos_history) > self.p_history_max_length):
            self.p_pos_history.pop(0)


class DroneState(EntityState):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # communication utterance
        #self.c = None
        # angle of antenna relative drone body (relative global coords)
        self.theta = None


class Action:
    def __init__(self):
        #communication action - if sending and direction - should the direction (omega) be in its own variable???
        self.c = None

class DroneAction(Action):  # action of the agent
    def __init__(self):
        super().__init__()
        # physical action
        self.u = None # controlls antenna
        #maybe only sets velocity directly and not acceleration?

        # deleting message from message buffer
        self.d = None

        # not used!
        self.communication = None #int taken from discrete action space with size of massege_buffer+1 (0 == no communication)

    
    def __str__(self):
        return f'Physical:{self.u}, Comm: {self.c}, {self.d}'


class Entity:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ""
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        #self.collide = True
        # material density (affects mass)
        #self.density = 25.0
        # color - only for rendering
        self.color = None
        self.color_intensity = 0.0 # the intensity of the rendered color - to fade out after transmission

        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        #self.initial_mass = 1.0
        # if the agent can(not) observe the world
        self.blind = False
        # the agent cannot send communication (different from self.c i state/action??)
        self.silent = False

        self.transmit_power = 1 #0.5625 # in SNR calculation

        self.current_jamming_factor = 0.

        self.internal_noise = 1 # internal noise for SNR calculation

        self.message_buffer = [] # enteties that communicate all have a storage of messages

        # the mesage to be sent by the entity
        self.message_to_send = None #[1,0,0,0,[self.name]]

    def get_index(self):
        """
        Returns the index corresponding to the entity, the name is on the form: name_{i}
        """
        return int(self.name.split("_")[1])
    
    # to easily see which base it is when printed out in terminal
    def __str__(self): 
        return f'{self.name}'
    
    # to easily see which base it is when printed out in terminal
    def __repr__(self):
        return f'{self.name}'

    #@property
    #def mass(self):
    #    return self.initial_mass

# Obs bases should listen aswell
class Base(Entity):  # properties of Base entities
    def __init__(self):
        super().__init__()
        self.action = Action() # ska detta vara en action - kommer inte tränas??

        self.generate_messages = True
    
    def __str__(self):
        return super().__str__()
    
    def __repr__(self):
        return super().__repr__()

class Emitter(Entity): # always transmitting - taking no actions (atleast yet)
    def __init__(self):
        super().__init__()
        self.state.c = True # always communicating
        self.blind = True # the emitters do not listen
        self.movable = True
        self.action = DroneAction()

    def check_boundary(base_positions, sample_position, transmission_radius):
        """
            Return False if jammer is out of bounds.
            Otherwise False
        """
        out_of_bounds = True
        if (base_positions[0][0] < sample_position[0] < base_positions[1][0] and
            -1.5 * transmission_radius < sample_position[1] < 1.5 * transmission_radius):
            # Within rectangle
            out_of_bounds = False

        for base_position in base_positions:
            if (np.linalg.norm(base_position - sample_position) < 1.5 * transmission_radius):
                out_of_bounds = False

        return out_of_bounds

    def generate_action(self, R):
        # Rejection sampling
        while self.action.u is None or np.linalg.norm(self.action.u) == 0:
            
            towards_center = np.array([R / 2, 0]) - self.state.p_pos
            # No idea why it inverts at some point..
            self.action.u = - towards_center / np.linalg.norm(towards_center)
            direction_offset = np.random.uniform( -math.pi / 2, math.pi / 2)
            #print("direction offset: ", direction_offset)
            rotation_matrix = np.array([[np.cos(direction_offset), - np.sin(direction_offset)],
                                        [np.sin(direction_offset) , np.cos(direction_offset)]])

            self.action.u = rotation_matrix @ self.action.u

        else:
            # Reverse direction
            self.action.u = self.action.u*-1
    def __str__(self):
        return super().__str__()
    
    def __repr__(self):
        return super().__repr__()


class Drone(Entity):  # properties of agent entities
    def __init__(self):
        super().__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        #self.u_noise = None
        # communication noise amount - to be changed!! also direction!
        self.c_noise = None
        # control range - what should we have in our problem?
        self.u_range = 1.0

        # state
        self.state = DroneState()
        #self.state.p_pos = None # add it here to try to solve bug were all agents' positions have the same id

        # action
        self.action = DroneAction()
        # script behavior to execute
        self.action_callback = None

        # the number of messages able to be stored at once
        self.message_buffer_size = 4
        # a list of the possible messages - list of lists (messages)
        self.message_buffer = []#np.zeros([self.message_buffer_size, 5]) 
        self.reward_bonus = 0

        self.movement_cost = 0.5 # the cost of movement - scales with magnitude of movement 
        self.radar_cost = 0.02 # cost of changing direction of radar
        self.transmission_cost = 0.001 # cost of transmitting a message 


    def __str__(self):
        return super().__str__()
    
    def __repr__(self):
        return super().__repr__()


class World:  # multi-agent world
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = [] # OBS change to drones later?? or keep as agents?
        self.bases = []
        self.emitters = []
        self.base_positions = None
        self.transmission_radius = None
        self.R = None # Distance between bases
        # communication channel dimensionality (we only have destination??)
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality - 
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # noice level in control signals - maybe attributes of the Drones instead? not the world?
        self.sigma_x = 0.0
        self.sigma_y = 0.0
        self.sigma_omgea = 0.0
        # physical damping
        #self.damping = 0.25
        # contact response parameters - not used by us?
        #self.contact_force = 1e2
        #self.contact_margin = 1e-3

        self.message_ID_counter = None # contains key(message_id): [destination, timestep for transmission]

    # return all entities in the world
    @property
    def entities(self):
        return self.bases + self.emitters + self.agents # the agents are last here so that they are drawn last
    
    # return all agents in the world
    @property
    def all_agents(self):
        return self.agents

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts - kan användas senare för störande drönare
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world 
    def step(self):
        # set actions for scripted agents - could be used later
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        for emitter in self.emitters:
            if Emitter.check_boundary(self.base_positions, emitter.state.p_pos, self.transmission_radius):
                emitter.generate_action(self.R)

        for i, agent in enumerate(self.agents):
            # used to check correct init (uniqe) id's for all agents
            #print(f"Agent {agent.name} p_pos id: {id(agent.state.c)}")
            self.apply_process_model_2_drones(agent, agent.action.u[:2], agent.action.u[2])

        for emitter in self.emitters:
            #velocity = np.array([1.0, 1.0])
            theta = 0.
            self.apply_process_model_2_drones(emitter, emitter.action.u[:2], theta)

    # NOT USED?
    # applies the process model from the pdf
    def apply_process_model(self):
        for i, entity in enumerate(self.entities):
            if not entity.movable: # skip enteties that can't move - currently bases and emitters 
                continue
            entity.state.p_pos += entity.state.p_vel * self.dt

            ## add acceleration part of the process model - only for drones
            if entity is Drone: #OBS check if this works!!! not sure (nor errors atleast)
                entity.state.p_pos += entity.action.u[:2] * self.dt**2/2 # here u is assumed to contain [a,a,omega]
                entity.state.p_vel += entity.action.u[:2] * self.dt # u borde nog innehålla acc och omega - som i overleafen
                # OBS ÄNDRA SÅ ATT u LÄGGS IN RÄTT u = [a, a, omega] - fixat

                entity.state.theta += entity.action.u[2] * self.dt
                #ensures theta is bounded between 0 and 2pi
                entity.state.theta %= (2*np.pi)

            ## here we can add the stochastic control noice 

            # ensure that max-speed is enforced
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                        entity.state.p_vel / np.sqrt(
                            np.square(entity.state.p_vel[0])
                            + np.square(entity.state.p_vel[1])
                        ) * entity.max_speed)
                    

    # TODO - uppdatera så att fiendens state också uppdateras varje tidsteg - här eller egen funktion.
    # the simpler model without acceleration - here action is setting the new velocity (and omega)
    def apply_process_model_2_drones(self, agent, velocity, theta):
        """
        Applies the physical transition kernel. 
        Steps all agents' position and orientation one time step.
        """
        if not agent.movable: # skip enteties that can't move - currently bases and emitters 
            return
        
        # assumes velocity is set immediatly - realistic?
        #print(agent.action.u)
        agent.state.p_vel = velocity # u[:2] contains velocity in x, y directions

        # stochastic control noise (gaussian)
        noise_scale = (np.array([self.sigma_x, self.sigma_y]) * agent.state.p_vel * self.dt)**2
        agent.state.p_vel += np.random.normal(loc = 0, scale = noise_scale, size = (2,)) 
        
        theta_action_dt = theta #* self.dt testing without the dt param - for discrete case 
        theta_noise = np.random.normal(loc = 0, scale = (self.sigma_omgea*theta_action_dt)**2)
        agent.state.theta += theta_action_dt + theta_noise
        #ensures that theta is bounded in [0,2pi)
        agent.state.theta %= (2*np.pi)


        # ensure that max-speed is enforced
        # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9161257
        if agent.max_speed is not None:
            speed = np.sqrt(np.square(agent.state.p_vel[0]) + np.square(agent.state.p_vel[1]))
            if speed > agent.max_speed:
                agent.state.p_vel = (
                    agent.state.p_vel / np.sqrt(
                        np.square(agent.state.p_vel[0])
                        + np.square(agent.state.p_vel[1])
                    ) * agent.max_speed)
        
        ## now the position is updated
        agent.state.p_pos += agent.state.p_vel * self.dt 


    ## updates the communication states of all enteties       
    def update_entity_state(self, entity):
        if entity.silent:
            entity.state.c = np.zeros(self.dim_c)
        else:
            #OBS add some stochastic noise or something similar. 
            # also remember transmitters/bases transmit in all directions, drones have direction
            noise = 0 # OBS should be stochastic
            entity.state.c = entity.action.c + noise 
        
"""
    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                p_force[i] = agent.action.u + noise
        return p_force



    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_pos += entity.state.p_vel * self.dt
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                        entity.state.p_vel
                        / np.sqrt(
                            np.square(entity.state.p_vel[0])
                            + np.square(entity.state.p_vel[1])
                        )
                        * entity.max_speed
                    )

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

"""


"""   
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent) # could be usefull to use something similar!!
        """