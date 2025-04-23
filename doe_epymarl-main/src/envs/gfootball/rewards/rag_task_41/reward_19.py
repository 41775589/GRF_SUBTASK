import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for attack skill enhancement by promoting finishing and clever offensive plays."""

    def __init__(self, env):
        super().__init__(env)
        self._num_zones = 5  # Number of zones towards the opponent's goal to track
        self._zone_rewards = np.linspace(0.05, 0.25, self._num_zones)  # Gradual increase in reward
        self._zone_thresholds = np.linspace(0.2, 1.0, self._num_zones)  # Thresholds for each zone
        self._collected_zones = [False] * self._num_zones
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and clear collected zones."""
        self._collected_zones = [False] * self._num_zones
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Compute the reward for an agent's actions considering ball movement and position."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        components = {"base_score_reward": reward,
                      "offensive_play_reward": 0.0}

        ball_x = observation['ball'][0]
        owned_team = observation['ball_owned_team']

        # Check if our team owns the ball and is advancing towards the goal
        if owned_team == 1:  # Assuming our agent is on the right team (team 1)
            for i in range(self._num_zones):
                if ball_x > self._zone_thresholds[i] and not self._collected_zones[i]:
                    reward += self._zone_rewards[i]
                    self._collected_zones[i] = True
                    components["offensive_play_reward"] += self._zone_rewards[i]

        return reward, components

    def step(self, action):
        """ Execute a step in the environment, calculate rewards, and return experience tuple."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = reward
        for key, value in components.items():
            info[f"component_{key}"] = value

        # Update sticky actions stats
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
