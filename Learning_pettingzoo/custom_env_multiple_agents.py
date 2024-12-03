import functools

import random
from copy import copy

import gymnasium
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv


"""
A masked verion of the same env
"""
class CustomActionMaskedEnvironment(ParallelEnv):
    """The metadata holds environment constants.

    The "name" metadata allows the environment to be pretty printed.
    """

    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self, render_mode=None, num_prisoners=2):
        """Initialize the environment with a configurable number of prisoner agents."""
        self.escape_y = None
        self.escape_x = None
        self.guard_y = None
        self.guard_x = None
        self.prisoner_positions = []  # To hold multiple prisoner positions
        self.timestep = None
        self.possible_agents = ["guard"]  # No prisoner agents at initialization
        self.render_mode = render_mode
        self.grid_size = None
        self.num_prisoners = num_prisoners  # Number of prisoner agents
        self.terminated_prisoners = set()

    def reset(self, seed=None, options=[7]):
        """Reset the environment with multiple prisoner agents."""
        self.agents = copy(self.possible_agents) + [f"prisoner_{i}" for i in range(self.num_prisoners)]
        self.timestep = 0
        self.grid_size = options[0]

        # Initial positions for the prisoner agents
        self.prisoner_positions = [(0, 0) for _ in range(self.num_prisoners)]  # All prisoners start at (0, 0)

        # Guard starts in bottom right corner
        self.guard_x = self.grid_size - 1
        self.guard_y = self.grid_size - 1

        # Random escape position
        self.escape_x = random.randint(2, self.grid_size - 2)
        self.escape_y = random.randint(2, self.grid_size - 2)

        # Observation setup
        observation = (
            [self.prisoner_positions[i][0] + self.grid_size * self.prisoner_positions[i][1] for i in range(self.num_prisoners)],
            self.guard_x + self.grid_size * self.guard_y,
            self.escape_x + self.grid_size * self.escape_y,
        )
        
        # Create action masks for each prisoner
        prisoner_action_masks = {f"prisoner_{i}": np.array([0, 1, 0, 1]) for i in range(self.num_prisoners)}
        prisoner_observations = {f"prisoner_{i}": {"observation": observation[0][i], "action_mask": prisoner_action_masks[f"prisoner_{i}"]} for i in range(self.num_prisoners)}

        observations = prisoner_observations
        observations.update({"guard": {"observation": observation[1], "action_mask": np.array([1, 0, 1, 0])}})

        infos = {a: {} for a in self.agents}

        return observations, infos
    
    def all_keys_equal_to(self, d, value): # used to check if all keys in dict have the same value, ex all agents terminate
        return all(v == value for v in d.values())

    def step(self, actions):
        """Step function with multiple prisoner agents."""
        prisoner_actions = {}
        for i in range(self.num_prisoners):
            prisoner_actions[f"prisoner_{i}"] = actions.get(f"prisoner_{i}", -1)  # Default to -1 if no action is taken
        
        guard_action = actions["guard"]

        # Update positions based on actions
        for i in range(self.num_prisoners):
            if f"prisoner_{i}" in self.terminated_prisoners: # escaped prisoners stop moving
                continue
            prisoner_action = prisoner_actions[f"prisoner_{i}"]
            if prisoner_action == 0 and self.prisoner_positions[i][0] > 0:
                self.prisoner_positions[i] = (self.prisoner_positions[i][0] - 1, self.prisoner_positions[i][1])
            elif prisoner_action == 1 and self.prisoner_positions[i][0] < self.grid_size - 1:
                self.prisoner_positions[i] = (self.prisoner_positions[i][0] + 1, self.prisoner_positions[i][1])
            elif prisoner_action == 2 and self.prisoner_positions[i][1] > 0:
                self.prisoner_positions[i] = (self.prisoner_positions[i][0], self.prisoner_positions[i][1] - 1)
            elif prisoner_action == 3 and self.prisoner_positions[i][1] < self.grid_size - 1:
                self.prisoner_positions[i] = (self.prisoner_positions[i][0], self.prisoner_positions[i][1] + 1)

        # Handle guard actions
        if guard_action == 0 and self.guard_x > 0:
            self.guard_x -= 1
        elif guard_action == 1 and self.guard_x < self.grid_size - 1:
            self.guard_x += 1
        elif guard_action == 2 and self.guard_y > 0:
            self.guard_y -= 1
        elif guard_action == 3 and self.guard_y < self.grid_size - 1:
            self.guard_y += 1

        # Generate action masks for prisoners after movement
        prisoner_action_masks = {}
        for i in range(self.num_prisoners):
            prisoner_action_mask = np.ones(4, dtype=np.int8)
            px, py = self.prisoner_positions[i]
            if px == 0:
                prisoner_action_mask[0] = 0
            elif px == self.grid_size - 1:
                prisoner_action_mask[1] = 0
            if py == 0:
                prisoner_action_mask[2] = 0
            elif py == self.grid_size - 1:
                prisoner_action_mask[3] = 0
            prisoner_action_masks[f"prisoner_{i}"] = prisoner_action_mask

        guard_action_mask = np.ones(4, dtype=np.int8)
        if self.guard_x == 0:
            guard_action_mask[0] = 0
        elif self.guard_x == self.grid_size - 1:
            guard_action_mask[1] = 0
        if self.guard_y == 0:
            guard_action_mask[2] = 0
        elif self.guard_y == self.grid_size - 1:
            guard_action_mask[3] = 0
            
        # Action mask to prevent guard from going over escape cell
        if self.guard_x - 1 == self.escape_x and self.guard_y == self.escape_y: 
            guard_action_mask[0] = 0
        elif self.guard_x + 1 == self.escape_x and self.guard_y == self.escape_y:
            guard_action_mask[1] = 0
        if self.guard_y - 1 == self.escape_y and self.guard_x == self.escape_x:
            guard_action_mask[2] = 0
        elif self.guard_y + 1 == self.escape_y and self.guard_x == self.escape_x:
            guard_action_mask[3] = 0


        # Termination conditions
        terminations = {a: False for a in self.agents}
        rewards = {a: 0 for a in self.agents}
        for i in range(self.num_prisoners):
            if self.prisoner_positions[i] == (self.guard_x, self.guard_y):
                self.terminated_prisoners.add(f"prisoner_{i}")
                rewards[f"prisoner_{i}"] = -1
                rewards["guard"] += 1 # fix reward to be num captured
                terminations[f"prisoner_{i}"] = True
                #terminations["guard"] = True
                #self.agents = []  # End the game for everyone

        #if any([pos == (self.escape_x, self.escape_y) for pos in self.prisoner_positions]):
            for i in range(self.num_prisoners):
                if self.prisoner_positions[i] == (self.escape_x, self.escape_y):
                    self.terminated_prisoners.add(f"prisoner_{i}")
                    rewards[f"prisoner_{i}"] = 1
                    rewards["guard"] -= 1 # fix reward - maybe works now
                    terminations[f"prisoner_{i}"] = True
            #self.agents = []

        ## check if all agents have escaped:
        if self.num_prisoners == len(self.terminated_prisoners): #obs: + captured prisoners
            self.agents = []
            terminations["guard"] = True
            #rewards["guard"] = -self.num_prisoners



        # Truncation conditions - has priority over termination!! has to think about this when assigning rewards
        truncations = {"prisoner_" + str(i): False for i in range(self.num_prisoners)}
        truncations["guard"] = False
        if self.timestep > 100:
            rewards = {"prisoner_" + str(i): 0 for i in range(self.num_prisoners)}
            rewards["guard"] = 0
            truncations = {"prisoner_" + str(i): True for i in range(self.num_prisoners)}
            truncations["guard"] = True
            self.agents = []

        self.timestep += 1

        # Generate new observations
        observations = {
            f"prisoner_{i}": {
                "observation": self.prisoner_positions[i][0] + self.grid_size * self.prisoner_positions[i][1],
                "action_mask": prisoner_action_masks[f"prisoner_{i}"],
            }
            for i in range(self.num_prisoners)
        }
        observations.update({"guard": {"observation": self.guard_x + self.grid_size * self.guard_y, "action_mask": guard_action_mask}})

        infos = {"prisoner_" + str(i): {} for i in range(self.num_prisoners)}
        infos.update({"guard": {}})

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""

        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        grid = np.full((self.grid_size, self.grid_size), "( )")
        for i in range(self.num_prisoners):
            if f"prisoner_{i}" in self.terminated_prisoners:
                grid[self.prisoner_positions[i][0], self.prisoner_positions[i][1]] = "(X)"
            else: 
                grid[self.prisoner_positions[i][0], self.prisoner_positions[i][1]] = "(P)"
        grid[self.guard_y, self.guard_x] = "(G)"
        grid[self.escape_y, self.escape_x] = "(E)"
        print(f"{grid} \n")

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        return MultiDiscrete([self.grid_size * self.grid_size - 1] * 3)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(4)