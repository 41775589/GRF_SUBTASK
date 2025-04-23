import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for shooting accuracy and power from central field positions."""

    def __init__(self, env):
        super().__init__(env)
        self.shooting_distance_threshold = 0.5  # Approx central field region
        self.shooting_power_reward = 0.2
        self.accuracy_multiplier = 5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and the sticky action counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the inherent environment state and add additional data if necessary."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the inherent environment state from saved data."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify the reward based on the ball's position and shooting actions.

        Args:
            reward (list[float]): The original rewards for each agent.

        Returns:
            tuple[list[float], dict[str, list[float]]]: Tuple containing the modified reward list and a dictionary of reward components.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_accuracy_reward": [0.0] * len(reward),
                      "shooting_power_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos_x = o['ball'][0]  # Extract X-coordinate of the ball
            ball_pos_y = o['ball'][1]  # Extract Y-coordinate of the ball

            # Check if shooting action is performed in central field region
            if self._is_central_zone(ball_pos_x, ball_pos_y):
                if 'ball_direction' in o and np.linalg.norm(o['ball_direction'][:2]) > 0:
                    # Calculate reward for shooting power based on ball speed
                    components["shooting_power_reward"][rew_index] = self.shooting_power_reward * np.linalg.norm(o['ball_direction'][:2])
                    
                    # Calculate reward for shooting accuracy based on closeness to goal
                    components["shooting_accuracy_reward"][rew_index] = self._calculate_accuracy_bonus(ball_pos_x, ball_pos_y)
                    
                # Sum up the additional shooting rewards to the base reward
                reward[rew_index] += (components["shooting_power_reward"][rew_index] + components["shooting_accuracy_reward"][rew_index])
        
        return reward, components

    def _is_central_zone(self, x, y):
        """Check if given coordinates are in the central shooting zone."""
        return abs(x) < self.shooting_distance_threshold and -0.42 < y < 0.42

    def _calculate_accuracy_bonus(self, x, y):
        """Calculate a bonus for accuracy based on how close to the goal center the shot is projected."""
        center_goal_y = 0.0  # Center of the goal on the y-axis
        distance_to_goal_center_y = abs(y - center_goal_y)
        # Inverse relationship: closer to center, higher the reward, up to a limit at the goal center
        return max(0, (self.accuracy_multiplier * (0.42 - distance_to_goal_center_y)))

    def step(self, action):
        """Collect rewards from the environment, process them with custom reward logic, then return results."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
