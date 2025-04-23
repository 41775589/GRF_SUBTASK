import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds precision shooting rewards near the opponent’s goal."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._num_zones = 5  # Divides close range of goal into 5 zones
        self._zone_reward = 0.1  # Reward for being in a new zone
        self._visited_zones = [False] * 5  # Track visited zones
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and zone tracking."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._visited_zones = [False] * 5
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._visited_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore state."""
        from_pickle = self.env.set_state(state)
        self._visited_zones = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Custom reward function focusing on accuracy and positioning near the goal."""
        observation = self.env.unwrapped.observation()  # Get the raw observations
        components = {"base_score_reward": reward.copy(), "precision_shooting_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # The position of the ball on the x-axis (1 is at opponent's goal)
            ball_x_position = o['ball'][0] 

            # Only consider the close range to the opponent's goal (last 0.2 on x-axis)
            if ball_x_position > 0.8:
                # Determine the current zone based on the x position
                zone_index = min(int((ball_x_position - 0.8) * 5 / 0.2), 4)  
                if not self._visited_zones[zone_index]:
                    self._visited_zones[zone_index] = True
                    components["precision_shooting_reward"][rew_index] = self._zone_reward
                    reward[rew_index] += self._zone_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
