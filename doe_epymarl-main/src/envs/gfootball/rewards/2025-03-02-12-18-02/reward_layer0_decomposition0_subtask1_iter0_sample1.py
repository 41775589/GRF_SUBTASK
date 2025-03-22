import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for defensive actions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._defensive_actions = ['Sliding', 'Stop-Dribble', 'Stop-Sprint']
        self._checkpoint_reward = 0.1

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, act in enumerate(observation['actions']):
            if act in self._defensive_actions:
                components["checkpoint_reward"][rew_index] = self._checkpoint_reward
                reward[rew_index] += self._checkpoint_reward

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        return observation, reward, done, info
