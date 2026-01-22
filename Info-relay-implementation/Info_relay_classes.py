from copy import copy
import math
import numpy as np
import csv
import pickle

from dataclasses import dataclass, field


class EvaluationLogger:
    def __init__(self, directed_transmission = False, jammer_on = False, K = 0, scenario_file = "", evaluation_log = "evaluation_log"):
        self.episode_index = 0
        self.success = False
        self.R = 0
        self.value = 0
        self.budget = 0
        self.episode_movement = 0
        self.episode_air_distance = 0
        self.delivery_time = 0
        self.directed_transmission = directed_transmission
        self.K = K
        self.scenario_file = scenario_file.split("/")[-1]
        self.evaluation_log = evaluation_log # change save path
        
        # idx, success, R, value, budget, sum_distance, air_distance, delivery_time, directed_transmission_bool, K, file
        self.f_eval = open(self.evaluation_log, 'w')
        self.writer_eval = csv.writer(self.f_eval)
        self.writer_eval.writerow(["idx", "success", "R", "value", "budget", "agent_sum_distance",
                                   "air_distance", "delivery_time", "directed_transmission", "K", "file"])

        # logging trajectories
        self.p_trajectories = {i: {} for i in range(K)}
        self.phi_trajectories = {i: {} for i in range(K)}

        # To save all data in all episodes
        self.episodes_data = {}  # {episode_idx: {timestep: {column_name: value}}}

        if self.evaluation_log.endswith(".csv"):
            self.pkl_path = self.evaluation_log[:-4] + "_1.pkl"
        else:
            self.pkl_path = self.evaluation_log + "_1.pkl"

           
    def log_trajectory(self, t, agents):
        for i, agent in enumerate(agents):
            pos = np.array(agent.state.p_pos, dtype=float)
            theta = getattr(agent.state, "theta", None)

            self.p_trajectories[i][t] = pos
            self.phi_trajectories[i][t] = theta

    def write_episode(self):
        self.writer_eval.writerow([self.episode_index,
                                   self.success,
                                   self.R,
                                   self.value,
                                   self.budget,
                                   self.episode_movement,
                                   self.episode_air_distance,
                                   self.delivery_time,
                                   self.directed_transmission,
                                   self.K,
                                   self.scenario_file])
        
        self.R = 0
        self.value = 0
        self.budget = 0
        self.episode_movement = 0
        self.episode_air_distance = 0
        self.delivery_time = 0
        self.success = False

        self.p_trajectories = {i: {} for i in range(self.K)}
        self.phi_trajectories = {i: {} for i in range(self.K)}

    def update_episode_index(self, index):
        self.episode_index = index

    def add_delivery_time(self, time):
        self.delivery_time += time

    def set_delivery_time(self, tot_time):
        self.delivery_time = tot_time

    def set_budget(self, budget):
        self.budget = budget

    def set_R(self, R):
        self.R = R

    def set_success(self):
        self.success = True

    def add_movement(self, movement):
        self.episode_movement += movement
    
    def set_movement(self, tot_movement):
        self.episode_movement = tot_movement

    def add_air_distance(self, sender, receiver):
        distance = np.linalg.norm(sender.state.p_pos - receiver.state.p_pos)
        self.episode_air_distance += distance

    def add_value(self, time_step, reward):
        self.value += reward * 0.99 ** time_step

    def set_value(self, value):
        self.value = value

    # logging all timesteps file
    def begin_episode(self, episode_idx):
        self.episode_index = episode_idx
        self.episodes_data[episode_idx] = {}  

    def log_step(self, t, agents, jammer=None, jammer_on=False):
        #Log one timestep of an episode.
        step_dict = {
            "idx": int(self.episode_index),
            "t": int(t),
            "R": self.R,
            "directed_transmission": self.directed_transmission,
            "jammer_on": jammer_on
        }

        for i, agent in enumerate(agents):
            step_dict[f"agent{i}_x"] = float(agent.state.p_pos[0])
            step_dict[f"agent{i}_y"] = float(agent.state.p_pos[1])
            step_dict[f"agent{i}_phi"] = float(getattr(agent.state, "theta", 0.0))
            step_dict[f"agent{i}_has_message"] = bool(agent.message_buffer)

        if jammer_on:
            jammer = jammer[0]
            step_dict["jammer_x"] = float(jammer.state.p_pos[0])
            step_dict["jammer_y"] = float(jammer.state.p_pos[1])
        else:
            step_dict["jammer_x"] = None
            step_dict["jammer_y"] = None
        

        self.episodes_data[self.episode_index][t] = step_dict


    def switch_file(self):
        """Switch from _1.pkl to _2.pkl."""
        self.pkl_path = self.pkl_path[:-5] + "2.pkl"
        self.episodes_data = {}
        print(f"Switched logging to: {self.pkl_path}")


    def save_episodes(self):
        """Save the nested dict to disk using pickle."""
        with open(self.pkl_path, "wb") as f:
            pickle.dump(self.episodes_data, f)


class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None 
        self.p_pos_history = []
        self.p_history_max_length = 200
        # physical velocity
        self.p_vel = None
        # communication utterance
        self.c = None 
        # transmitting if True, listening if False. 

    def save_history(self):
        self.p_pos_history.append(copy(self.p_pos))
        while (len(self.p_pos_history) > self.p_history_max_length):
            self.p_pos_history.pop(0)


class DroneState(EntityState):  # state of agents (including communication)
    def __init__(self):
        super().__init__()
        # angle of antenna relative drone body (relative global coords)
        self.theta = None


class Action:
    def __init__(self):
        #communication action 
        self.c = None

class DroneAction(Action):  # action of the agent
    def __init__(self):
        super().__init__()
        # physical action
        self.u = None # controlls antenna

        # deleting message from message buffer
        self.d = None

    
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
        
        # if the agent can(not) observe the world
        self.blind = False
        # the agent cannot send communication 
        self.silent = False

        self.transmit_power = 1 

        self.current_jamming_factor = 0.0

        self.internal_noise = 1 # internal noise for SNR calculation

        self.message_buffer = None # enteties that communicate all have a storage of messages


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


class Base(Entity):  # properties of Base entities
    def __init__(self):
        super().__init__()
        self.action = Action()

        self.generate_messages = True
    
    def __str__(self):
        return super().__str__()
    
    def __repr__(self):
        return super().__repr__()

class Emitter(Entity): 
    def __init__(self):
        super().__init__()
        self.state.c = True # always communicating
        self.blind = True 
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
            
            self.action.u = - towards_center / np.linalg.norm(towards_center)
            direction_offset = np.random.uniform( -math.pi / 2, math.pi / 2)
            
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
        # control range
        self.u_range = 1.0

        # state
        self.state = DroneState()

        # action
        self.action = DroneAction()
        # script behavior to execute
        self.action_callback = None

        # the number of messages able to be stored at once
        self.message_buffer_size = 1
        
        self.message_buffer = None
        self.reward_bonus = 0

        self.movement_cost = 0.5 # the cost of movement - scales with magnitude of movement 
        self.radar_cost = 0.1 # cost of changing direction of radar
        #self.transmission_cost = 0.001 # cost of transmitting a message 


    def __str__(self):
        return super().__str__()
    
    def __repr__(self):
        return super().__repr__()


class World: # multi-agent world
    def __init__(self):
        # list of agents and entities 
        self.agents = [] 
        self.bases = []
        self.emitters = []
        self.base_positions = None
        self.transmission_radius = None
        self.R = None # Distance between bases
        # communication channel dimensionality 
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # noice level in control signals
        self.sigma_x = 0.0
        self.sigma_y = 0.0
        self.sigma_omgea = 0.0

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

    # return all agents controlled by world scripts
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
            self.apply_process_model(agent, agent.action.u[:2], agent.action.u[2])

        for emitter in self.emitters:
            theta = 0.
            self.apply_process_model(emitter, emitter.action.u[:2], theta)
                    

    # the process model where the actions are instantaneous velocities 
    def apply_process_model(self, agent, velocity, theta):
        """
        Applies the physical transition kernel. 
        Steps all agents' position and orientation one time step.
        """
        if not agent.movable: # skip enteties that can't move
            return
        
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
        if agent.max_speed is not None:
            speed = np.sqrt(np.square(agent.state.p_vel[0]) + np.square(agent.state.p_vel[1]))
            if speed > agent.max_speed:
                agent.state.p_vel = (
                    agent.state.p_vel / np.sqrt(
                        np.square(agent.state.p_vel[0])
                        + np.square(agent.state.p_vel[1])
                    ) * agent.max_speed)
        
        # update the position with
        agent.state.p_pos += agent.state.p_vel * self.dt 


    ## updates the communication states of all enteties       
    def update_entity_state(self, entity):
        if entity.silent:
            entity.state.c = np.zeros(self.dim_c)
        else:
            noise = 0 # could be stochastic
            entity.state.c = entity.action.c + noise 


"""
    def apply_process_model_old(self):
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
    """
        