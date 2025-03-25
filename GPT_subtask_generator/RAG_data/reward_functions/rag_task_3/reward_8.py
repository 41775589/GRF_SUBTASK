import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense shooting practice reward into the Google Research Football environment."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define constants for different pressures situations
        self.low_pressure_threshold = 0.1    # distance from the nearest opponent considered as low pressure
        self.medium_pressure_threshold = 0.3 # distance from the nearest opponent considered as medium pressure
        self.shooting_distance_threshold = 0.6  # distance within which a shot is considered to be potentially scoring
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'pressure_reward': [0.0]}

        if observation is None:
            return reward, components

        ball_position = observation[0]['ball'][:2]  # ignore z-coordinate
        active_player_position = observation[0]['left_team'][observation[0]['active']]
        distances_to_opponents = np.linalg.norm(
            observation[0]['right_team'] - active_player_position, axis=1)
        min_distance_to_opponent = np.min(distances_to_opponents)

        # Determine the pressure level reward
        pressure_reward = 0
        if min_distance_to_opponent > self.medium_pressure_threshold:
            pressure_reward = 0.3  # High pressure
        elif min_distance_to_opponent > self.low_pressure_threshold:
            pressure_reward = 0.2  # Medium pressure
        else:
            pressure_reward = 0.1  # Low pressure
        
        # Check if goal post within a certain reasonable range to simulate shooting practice
        if (np.abs(ball_position[0] - 1) < self.shooting_distance_threshold):
            reward += 2  # potential goal scoring position
            components['pressure_reward'] = [pressure_reward]
            reward += pressure_reward

        return [reward], components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
