import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward for shots taken from a distance outside the penalty box, encouraging the practice of long-range shooting in the presence of opposing defenders."""

    def __init__(self, env):
        super().__init__(env)
        self.shooting_distance_threshold = 0.7  # Threshold to consider a long-range shot
        self.defender_proximity_threshold = 0.1  # How close a defender must be to consider the distance challenging
        self.long_range_shot_reward = 0.2  # Additional reward for taking a long-range shot
        self.defender_presence_reward = 0.3  # Additional reward for shooting near defenders
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset environment and clear any counters or holders."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Modify the reward based on the shooting conditions. Encourages shooting from distance and in challenging circumstances."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_range_shot_reward": 0.0,
                      "defender_presence_reward": 0.0}
        
        if observation is None:
            return reward, components

        ball_pos = observation['ball'][:2]  # Get only x, y position of the ball
        defenders_pos = observation['right_team']
        own_pos = observation['left_team'][observation['active']]
        
        # Check if shooting from a long distance
        if ball_pos[0] > self.shooting_distance_threshold:
            components["long_range_shot_reward"] = self.long_range_shot_reward
            reward += self.long_range_shot_reward

        # Check if any defender is close
        for defender in defenders_pos:
            if np.linalg.norm(defender - own_pos) < self.defender_proximity_threshold:
                components["defender_presence_reward"] = self.defender_presence_reward
                reward += self.defender_presence_reward
                break  # Only one reward per step, even if multiple defenders are close

        return reward, components

    def step(self, action):
        """Take a step using the given action and augment the observation-space with reward components."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = reward
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
