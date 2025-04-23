import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward to promote attacking and
       creative offensive play by placing emphasis on progressing towards the opponent's goal."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._num_zones = 10  # Divide the attacking half into 10 zones
        self._zone_reward = 0.05  # Reward for entering a new zone closer to the opponent's goal
        self._collected_zones = [0, 0, 0, 0]  # To track zones reached by each agent
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        self._collected_zones = [0, 0, 0, 0]
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "zone_reward": [0.0] * len(reward)}

        # Calculate rewards based on zone advancement
        for i in range(len(reward)):
            current_position = observation[i]['left_team'][observation[i]['active']]
            zone_id = int((current_position[0] + 1) * (self._num_zones / 2))
            if self._collected_zones[i] < zone_id:
                components["zone_reward"][i] = (zone_id - self._collected_zones[i]) * self._zone_reward
                reward[i] += components["zone_reward"][i]
                self._collected_zones[i] = zone_id

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        rewritten_reward, components = self.reward(reward)
        info["final_reward"] = sum(rewritten_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, act in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += act
        return observation, rewritten_reward, done, info
