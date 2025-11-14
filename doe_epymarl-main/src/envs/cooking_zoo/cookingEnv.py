

import sys
import os
import numpy as np
import gym
import torch as th
import cooking_zoo
from ..multiagentenv import MultiAgentEnv

from typing import Dict, List, Any, Tuple
from pettingzoo.utils import wrappers
from cooking_zoo.environment import cooking_env


class CookingZooEnv(MultiAgentEnv):
    """
    Cooking Zoo environment wrapper for EPyMARL
    """

    def __init__(self, **kwargs):
        """
        Initialize Cooking Zoo environment

        Args:
            level: str, cooking zoo level name (default: 'coop_test')
            num_agents: int, number of agents (default: 2)
            max_steps: int, maximum number of steps per episode (default: 400)
            obs_spaces: list, observation spaces for agents (default: ["feature_vector", "feature_vector"])
            action_scheme: str, action scheme (default: 'scheme3')
            meta_file: str, meta file name (default: 'example')
            recipes: list, recipes to cook (default: ["TomatoLettuceSalad"])
            end_condition_all_dishes: bool, end when all dishes complete (default: True)
            agent_visualization: list, agent visualization (default: ["human", "robot"])
            reward_scheme: dict, reward configuration (default: standard rewards)
            render: bool, whether to render (default: False)
        """
        # Extract arguments with defaults based on README
        self.level = kwargs.get('level', 'coop_test')
        self.n_agents = kwargs.get('num_agents', 2)
        self.episode_limit = kwargs.get('max_steps', 400)
        self.obs_spaces = kwargs.get('obs_spaces', ["feature_vector"] * self.n_agents)
        self.action_scheme = kwargs.get('action_scheme', 'scheme1')
        self.meta_file = kwargs.get('meta_file', 'example')
        self.recipes = kwargs.get('recipes', ["TomatoLettuceSalad"])
        self.end_condition_all_dishes = kwargs.get('end_condition_all_dishes', True)
        self.agent_visualization = kwargs.get('agent_visualization', ["human", "robot"])
        self.render_mode = kwargs.get('render', False)

        # Default reward scheme from README
        default_reward_scheme = {
            "recipe_reward": 50,
            "max_time_penalty": -1,
            "recipe_penalty": -20,
            "recipe_node_reward": 5
        }
        self.reward_scheme = kwargs.get('reward_scheme', default_reward_scheme)

        # Initialize episode tracking
        self.time_step = 0
        self.obs = None

        # Initialize the environment using cooking_env.parallel_env
        self.env = cooking_env.parallel_env(
            level=self.level,
            meta_file=self.meta_file,
            num_agents=self.n_agents,
            max_steps=self.episode_limit,
            recipes=self.recipes,
            agent_visualization=self.agent_visualization,
            obs_spaces=self.obs_spaces,
            end_condition_all_dishes=self.end_condition_all_dishes,
            action_scheme=self.action_scheme,
            render=self.render_mode,
            reward_scheme=self.reward_scheme
        )

        # Get observation and action space information
        self.observation_space = self.env.observation_space("player_0")
        self.action_space = self.env.action_space("player_0")


        # Initialize obs_size based on observation space
        self._setup_obs_size()

        # Set number of actions
        self.n_actions = self.action_space.n


    def _setup_obs_size(self):
        """Setup observation size based on observation space"""
        if hasattr(self.observation_space, 'shape'):
            if len(self.observation_space.shape) == 0:
                # Discrete observation space
                self.obs_size = 1
            else:
                # Box observation space - flatten all dimensions
                self.obs_size = int(np.prod(self.observation_space.shape))
        elif hasattr(self.observation_space, 'n'):
            # Discrete space
            self.obs_size = self.observation_space.n
        else:
            # Fallback - try to get from environment's feature vector length
            try:
                # Access the underlying CookingEnvironment
                cooking_env = self.env.env  # parallel_env wraps CookingEnvironment
                if hasattr(cooking_env, 'feature_vector_representation_length'):
                    self.obs_size = cooking_env.feature_vector_representation_length
                else:
                    self.obs_size = 64  # Default fallback
            except:
                self.obs_size = 64  # Default fallback

    def reset(self, **kwargs):
        """Reset the environment"""
        self.time_step = 0
        obs, info = self.env.reset()

        # Convert observations to numpy array and store
        obs_list = [obs[f"player_{i}"] for i in range(len(obs))]
        self.obs = np.array(obs_list)

        return obs_list, [info[f"player_{i}"] for i in range(len(info))]

    def step(self, actions):
        """Step the environment"""
        self.time_step += 1
        action_dict = {f"player_{i}": actions[i] for i in range(len(actions))}
        obs, reward, termination, truncation, info = self.env.step(action_dict)
        # print("rrrrrrr:",reward)

        # Update stored observations
        obs_list = [obs[f"player_{i}"] for i in range(len(obs))]
        self.obs = np.array(obs_list)

        # Process rewards (sum all agent rewards for EPyMARL)
        reward_list = [reward[f"player_{i}"] for i in range(len(reward))]
        total_reward = sum(reward_list)

        # Process termination (episode ends if any agent terminates or truncates)
        termination_list = [termination[f"player_{i}"] for i in range(len(termination))]
        truncation_list = [truncation[f"player_{i}"] for i in range(len(truncation))]
        terminated = any(termination_list) or any(truncation_list) or (self.time_step >= self.episode_limit)

        # Process info
        info_list = [info[f"player_{i}"] for i in range(len(info))]
        env_info = {
            'episode_limit': int(self.time_step >= self.episode_limit),
            'total_reward': total_reward,
            'n_terminated_agents': sum(termination_list),
            'n_truncated_agents': sum(truncation_list),
        }

        # Add any numeric info from individual agent infos
        for i, agent_info in enumerate(info_list):
            if isinstance(agent_info, dict):
                for key, value in agent_info.items():
                    if isinstance(value, (int, float, bool)):
                        # Create agent-specific keys for numeric values
                        env_info[f'agent_{i}_{key}'] = float(value)

        return total_reward, terminated, env_info

    def get_obs(self):
        """Returns all agent observations in a list."""
        if self.obs is not None:
            return self.obs.reshape(self.n_agents, -1)
        else:
            # Fallback: return zero observations
            return np.zeros((self.n_agents, self.obs_size))

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id."""
        obs_all = self.get_obs()
        return obs_all[agent_id].reshape(-1)

    def get_obs_size(self):
        """Returns the size of the observation."""
        return self.obs_size

    def get_state(self) -> np.ndarray:
        """Get global state (concatenated observations of all agents)"""
        return self.get_obs().flatten()

    def get_global_state(self) -> np.ndarray:
        """Alias for get_state for compatibility"""
        return self.get_state()

    def get_state_size(self) -> int:
        """Get size of global state"""
        return self.obs_size * self.n_agents

    def get_avail_actions(self) -> List[List[int]]:
        """Get available actions for each agent"""
        # In Cooking Zoo, all actions are typically available
        return [[1 for _ in range(self.n_actions)] for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id: int) -> List[int]:
        """Get available actions for specific agent"""
        return self.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return self.n_actions

    def get_stats(self) -> Dict:
        """Get environment statistics"""
        return {}

    def get_env_info(self) -> Dict:
        """Get environment information for EPyMARL"""
        return {
            'state_shape': self.get_state_size(),
            'obs_shape': self.get_obs_size(),
            'n_actions': self.get_total_actions(),
            'n_agents': self.n_agents,
            'episode_limit': self.episode_limit
        }

    def close(self):
        """Close the environment"""
        self.env.close()

    def seed(self, seed=None):
        """Set random seed (placeholder for compatibility)"""
        pass

    def render(self, mode='human'):
        """Render the environment"""
        self.env.render()

    def save_replay(self):
        """Save a replay (placeholder for compatibility)"""
        pass