"""
Base Reward Wrapper for CookingZoo
Provides common functionality for all custom reward wrappers.
"""

from pettingzoo.utils.wrappers import BaseWrapper
import numpy as np
from reward_wrapper_registry import register_reward_wrapper


class BaseRewardWrapper(BaseWrapper):
    """
    Base class for all custom reward wrappers.
    Provides common functionality and utilities.
    """

    def __init__(self, env, reward_config=None):
        """
        Args:
            env: CookingEnvironment instance
            reward_config: Dictionary containing reward function parameters
        """
        super().__init__(env)
        self.reward_config = reward_config or {}
        self.prev_global_state = None

    def reset(self, seed=None, return_info=False, options=None):
        """Reset environment and initialize state cache"""
        result = super().reset(seed=seed, return_info=return_info, options=options)
        self.prev_global_state = self._get_global_state()
        return result

    def step(self, action):
        """Override step to apply custom rewards"""
        # Store state before step
        prev_state = self._get_global_state()

        # Execute original step
        super().step(action)

        # Get new state after step
        current_state = self._get_global_state()

        # Only recompute rewards if this is the last agent's action
        # or if action is None (inactive agent)
        if hasattr(self.env, '_agent_selector'):
            if self.env._agent_selector.is_last() or action is None:
                self._recompute_rewards(prev_state, current_state)

        self.prev_global_state = current_state

    def _get_global_state(self):
        """
        Extract global state from the environment.
        Override this if you need additional state information.
        """
        world = self.env.world

        global_state = {
            'timestep': self.env.t,
            'max_steps': self.env.max_steps,
            'agent_positions': [agent.location for agent in world.agents],
            'agent_active': world.active_agents[:],
            'agent_holding': [agent.holding for agent in world.agents],
            'world_objects': dict(world.world_objects),
            'world_map': world.world_map,
            'recipe_graphs': self.env.recipe_graphs,
            'num_agents': len(world.agents),
        }

        return global_state

    def _recompute_rewards(self, prev_state, current_state):
        """
        Recompute rewards for all agents based on global state.
        This replaces the rewards computed by the base environment.
        """
        for agent_name in self.agents:
            if agent_name not in self.env.world_agent_to_env_agent_mapping.values():
                continue

            agent_idx = self.env.possible_agents.index(agent_name)

            # Compute custom reward
            custom_reward = self._compute_agent_reward(
                agent_idx,
                agent_name,
                prev_state,
                current_state
            )

            # Replace the reward
            self.rewards[agent_name] = custom_reward
            self._cumulative_rewards[agent_name] += custom_reward

    def _compute_agent_reward(self, agent_idx, agent_name, prev_state, current_state):
        """
        Main reward computation function.
        MUST be overridden by subclasses.

        Args:
            agent_idx: Index of the agent
            agent_name: Name of the agent (e.g., "player_0")
            prev_state: Global state before the action
            current_state: Global state after the action

        Returns:
            float: The computed reward
        """
        raise NotImplementedError("Subclasses must implement _compute_agent_reward")

    # Utility functions

    def _manhattan_distance(self, pos1, pos2):
        """Compute Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _euclidean_distance(self, pos1, pos2):
        """Compute Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def _get_agent_world_object(self, agent_idx, state):
        """Get the world agent object for a given agent index"""
        return self.env.world.agents[agent_idx]

    def _is_agent_active(self, agent_idx, state):
        """Check if an agent is active"""
        return state['agent_active'][agent_idx]

    def _get_objects_by_type(self, object_type, state):
        """Get all objects of a specific type from the world"""
        return state['world_objects'].get(object_type, [])

    def _get_nearest_object(self, agent_pos, object_type, state):
        """
        Find the nearest object of a given type to an agent.

        Returns:
            tuple: (object, distance) or (None, float('inf')) if no objects found
        """
        objects = self._get_objects_by_type(object_type, state)
        if not objects:
            return None, float('inf')

        min_dist = float('inf')
        nearest_obj = None

        for obj in objects:
            if hasattr(obj, 'location'):
                dist = self._manhattan_distance(agent_pos, obj.location)
                if dist < min_dist:
                    min_dist = dist
                    nearest_obj = obj

        return nearest_obj, min_dist

    def _count_agents_at_position(self, position, state):
        """Count how many agents are at a given position"""
        pos_tuple = tuple(position)
        return sum(1 for i, pos in enumerate(state['agent_positions'])
                   if tuple(pos) == pos_tuple and state['agent_active'][i])

    def _get_recipe_progress(self, agent_idx, state):
        """
        Get recipe completion progress for an agent.

        Returns:
            float: Progress between 0.0 and 1.0
        """
        if agent_idx >= len(state['recipe_graphs']):
            return 0.0

        recipe = state['recipe_graphs'][agent_idx]
        if hasattr(recipe, 'get_progress'):
            return recipe.get_progress()
        elif hasattr(recipe, 'completed'):
            return 1.0 if recipe.completed() else 0.0

        return 0.0


@register_reward_wrapper("default")
class DefaultRewardWrapper(BaseRewardWrapper):
    """
    Default reward wrapper that keeps the original reward structure
    but allows configuration.
    """

    def __init__(self, env, reward_config=None):
        default_config = {
            'recipe_reward': 20,
            'max_time_penalty': -5,
            'recipe_penalty': -40,
            'recipe_node_reward': 0
        }
        # Merge with provided config
        if reward_config:
            default_config.update(reward_config)
        super().__init__(env, default_config)

    def _compute_agent_reward(self, agent_idx, agent_name, prev_state, current_state):
        """Use the original reward computation"""
        # Keep original behavior
        reward = 0.0

        if agent_idx >= len(current_state['recipe_graphs']):
            return reward

        recipe = current_state['recipe_graphs'][agent_idx]
        prev_recipe = prev_state['recipe_graphs'][agent_idx]

        # Recipe completion
        if hasattr(recipe, 'completed'):
            if recipe.completed() and not prev_recipe.completed():
                reward += self.reward_config['recipe_reward']
            elif not recipe.completed() and prev_recipe.completed():
                reward += self.reward_config['recipe_penalty']

        # Time penalty
        reward += (self.reward_config['max_time_penalty'] /
                   current_state['max_steps'])

        return reward