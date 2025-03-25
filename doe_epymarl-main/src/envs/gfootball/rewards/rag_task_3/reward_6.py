import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds reward for practicing shots under pressure."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.target_zones = [
            {'x_range': (0.8, 1.0), 'y_range': (-0.2, 0.2), 'reward': 0.5},  # Close range central
            {'x_range': (0.7, 0.9), 'y_range': (0.2, 0.42), 'reward': 0.3},  # Right side of the box
            {'x_range': (0.7, 0.9), 'y_range': (-0.42, -0.2), 'reward': 0.3}   # Left side of the box
        ]
        self._shot_attempted = False

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._shot_attempted = False
        return self.env.reset()

    def reward(self, reward):
        """Rewards are provided based on the position from which the shot was taken and game mode pressures."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward

        o = observation[0]  # Considering single agent mode for simplicity

        # Detect if a shot was attempted
        if self.sticky_actions_counter[9] and o['ball_owned_team'] == 0:
            self._shot_attempted = True

        # Award additional reward if the shot was taken from a target zone and during game mode pressure
        if self._shot_attempted and o['ball_owned_team'] == -1:  # Shot taken and lost possession
            x, y = o['ball'][0], o['ball'][1]
            for zone in self.target_zones:
                if zone['x_range'][0] <= x <= zone['x_range'][1] and zone['y_range'][0] <= y <= zone['y_range'][1]:
                    reward = reward + zone['reward']
                    break
            self._shot_attempted = False  # Reset shot attempt tracker

        return reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = self.reward(reward)
        info["final_reward"] = sum(reward)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for i, action in enumerate(obs[0]['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
