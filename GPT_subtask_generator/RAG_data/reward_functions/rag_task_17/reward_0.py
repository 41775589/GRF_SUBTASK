import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward for mastering wide midfield responsibilities,
    focusing on high passes and positioning to expand the field of play and support lateral transitions.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._high_pass_reward = 0.5  # Additional reward for executing a high pass
        self._positioning_reward = 0.2  # Reward for positioning to stretch the opposition defense

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        # Prepare the components dictionary to hold individual rewards
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        # Access current observation
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        # Assuming observation is a dict including necessary elements for computation
        for i, o in enumerate(observation):
            if o['sticky_actions'][9] == 1:  # Checking if high pass action is active
                components["high_pass_reward"][i] = self._high_pass_reward
                reward[i] += components["high_pass_reward"][i] * 1.5

            # Positioning logic
            # Checking if the player is in the lateral positions of the field to extend play
            # player_position[i, 0] ranges from -1 to 1 where 1 is the right-most point of the field
            if abs(o['right_team'][o['active']][0]) < 0.2:  # lateral positions near the middle
                components["positioning_reward"][i] = self._positioning_reward
                reward[i] += components["positioning_reward"][i]

        return reward, components

    def step(self, action):
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
