"""
Cooperation-focused reward wrapper for CookingZoo.
Encourages agents to work together effectively.
"""

from base_wrapper import BaseRewardWrapper
from reward_wrapper_registey import register_reward_wrapper


@register_reward_wrapper("cooperation")
class CooperationRewardWrapper(BaseRewardWrapper):
    """
    Reward wrapper that emphasizes cooperation between agents.

    Reward components:
    - Recipe completion (base reward)
    - Cooperation bonus (agents at different stations)
    - Distance-based reward (agents staying apart)
    - Collision penalty (agents too close)
    - Idle penalty (not moving)
    """

    def __init__(self, env, reward_config=None):
        default_config = {
            # Base rewards
            'recipe_completion': 20.0,
            'recipe_penalty': -10.0,

            # Cooperation rewards
            'cooperation_bonus': 2.0,
            'distance_bonus': 0.5,
            'min_distance': 2.0,  # Minimum distance between agents for bonus

            # Penalties
            'collision_penalty': -1.0,
            'idle_penalty': -0.1,
            'time_penalty': -0.05,
        }

        if reward_config:
            default_config.update(reward_config)

        super().__init__(env, default_config)

    def _compute_agent_reward(self, agent_idx, agent_name, prev_state, current_state):
        """Compute reward with cooperation emphasis"""

        if not self._is_agent_active(agent_idx, current_state):
            return 0.0

        reward = 0.0

        # 1. Base recipe completion reward
        reward += self._recipe_completion_reward(agent_idx, prev_state, current_state)

        # 2. Cooperation bonus
        reward += self._cooperation_bonus(agent_idx, current_state)

        # 3. Distance-based reward
        reward += self._distance_reward(agent_idx, current_state)

        # 4. Collision penalty
        reward += self._collision_penalty(agent_idx, current_state)

        # 5. Idle penalty
        reward += self._idle_penalty(agent_idx, prev_state, current_state)

        # 6. Time penalty
        reward += self.reward_config['time_penalty']

        return reward

    def _recipe_completion_reward(self, agent_idx, prev_state, current_state):
        """Reward for completing recipes"""
        if agent_idx >= len(current_state['recipe_graphs']):
            return 0.0

        recipe = current_state['recipe_graphs'][agent_idx]
        prev_recipe = prev_state['recipe_graphs'][agent_idx]

        if hasattr(recipe, 'completed'):
            # Completed
            if recipe.completed() and not prev_recipe.completed():
                return self.reward_config['recipe_completion']
            # Failed/Reset
            elif not recipe.completed() and prev_recipe.completed():
                return self.reward_config['recipe_penalty']

        return 0.0

    def _cooperation_bonus(self, agent_idx, current_state):
        """
        Bonus when agents are working at different locations.
        Encourages division of labor.
        """
        agent_pos = tuple(current_state['agent_positions'][agent_idx])

        # Get positions of other active agents
        other_positions = [
            tuple(pos) for i, pos in enumerate(current_state['agent_positions'])
            if i != agent_idx and current_state['agent_active'][i]
        ]

        if not other_positions:
            return 0.0

        # Bonus if all agents at different positions
        all_positions = [agent_pos] + other_positions
        unique_positions = len(set(all_positions))

        if unique_positions == len(all_positions):
            return self.reward_config['cooperation_bonus']

        return 0.0

    def _distance_reward(self, agent_idx, current_state):
        """
        Reward for maintaining good distance from other agents.
        Prevents crowding.
        """
        agent_pos = current_state['agent_positions'][agent_idx]

        # Calculate minimum distance to other active agents
        min_distance = float('inf')
        for i, pos in enumerate(current_state['agent_positions']):
            if i != agent_idx and current_state['agent_active'][i]:
                dist = self._manhattan_distance(agent_pos, pos)
                min_distance = min(min_distance, dist)

        # Bonus if maintaining good distance
        if min_distance >= self.reward_config['min_distance']:
            return self.reward_config['distance_bonus']

        return 0.0

    def _collision_penalty(self, agent_idx, current_state):
        """Penalty for being too close to other agents"""
        agent_pos = current_state['agent_positions'][agent_idx]

        # Count agents at same or adjacent positions
        collision_count = 0
        for i, pos in enumerate(current_state['agent_positions']):
            if i != agent_idx and current_state['agent_active'][i]:
                dist = self._manhattan_distance(agent_pos, pos)
                if dist == 0:  # Same position
                    collision_count += 1

        return self.reward_config['collision_penalty'] * collision_count

    def _idle_penalty(self, agent_idx, prev_state, current_state):
        """Penalty for not moving"""
        prev_pos = prev_state['agent_positions'][agent_idx]
        curr_pos = current_state['agent_positions'][agent_idx]

        if prev_pos == curr_pos:
            return self.reward_config['idle_penalty']

        return 0.0


@register_reward_wrapper("sparse_cooperation")
class SparseCooperationRewardWrapper(CooperationRewardWrapper):
    """
    Sparse version of cooperation reward.
    Only gives rewards for major events (recipe completion).
    Useful for learning with less reward shaping.
    """

    def _compute_agent_reward(self, agent_idx, agent_name, prev_state, current_state):
        """Only reward recipe completion"""

        if not self._is_agent_active(agent_idx, current_state):
            return 0.0

        reward = 0.0

        # Only recipe completion reward
        reward += self._recipe_completion_reward(agent_idx, prev_state, current_state)

        # Minimal time penalty
        reward += self.reward_config['time_penalty']

        return reward


@register_reward_wrapper("dense_cooperation")
class DenseCooperationRewardWrapper(CooperationRewardWrapper):
    """
    Dense version with additional shaping rewards.
    Includes progress-based rewards.
    """

    def __init__(self, env, reward_config=None):
        default_config = {
            'recipe_completion': 20.0,
            'recipe_penalty': -10.0,
            'cooperation_bonus': 2.0,
            'distance_bonus': 0.5,
            'min_distance': 2.0,
            'collision_penalty': -1.0,
            'idle_penalty': -0.1,
            'time_penalty': -0.05,
            'progress_reward': 0.2,  # Additional reward for partial progress
            'pickup_reward': 0.5,
        }

        if reward_config:
            default_config.update(reward_config)

        BaseRewardWrapper.__init__(self, env, default_config)

    def _compute_agent_reward(self, agent_idx, agent_name, prev_state, current_state):
        """Include all cooperation rewards plus progress shaping"""

        # Get base cooperation rewards
        reward = super()._compute_agent_reward(
            agent_idx, agent_name, prev_state, current_state
        )

        # Add progress reward
        reward += self._progress_reward(agent_idx, prev_state, current_state)

        # Add pickup reward
        reward += self._pickup_reward(agent_idx, prev_state, current_state)

        return reward

    def _progress_reward(self, agent_idx, prev_state, current_state):
        """Reward for making progress on recipe"""
        prev_progress = self._get_recipe_progress(agent_idx, prev_state)
        curr_progress = self._get_recipe_progress(agent_idx, current_state)

        progress_delta = curr_progress - prev_progress

        if progress_delta > 0:
            return self.reward_config['progress_reward'] * progress_delta

        return 0.0

    def _pickup_reward(self, agent_idx, prev_state, current_state):
        """Small reward for picking up items"""
        prev_holding = prev_state['agent_holding'][agent_idx]
        curr_holding = current_state['agent_holding'][agent_idx]

        if prev_holding is None and curr_holding is not None:
            return self.reward_config['pickup_reward']

        return 0.0