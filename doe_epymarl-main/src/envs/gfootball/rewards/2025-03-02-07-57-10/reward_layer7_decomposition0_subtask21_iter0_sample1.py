import gym
import numpy as np
class CheckpointRewardWrapper(RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for improving offensive passing strategies and coordination in attack."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._collected_checkpoints = {0: 0, 1: 0}  # Initializing checkpoints for each agent
        self._num_checkpoints = 5
        self._checkpoint_reward = 0.1

    def reset(self):
        self._collected_checkpoints = {0: 0, 1: 0}  # Resetting checkpoints for each agent
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "checkpoint_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]  # Get observation for the agent

            # Check if the ball is close to the opponent's goal
            distance_to_goal = abs(o['ball'][0])  # Considering only X position as distance
            if distance_to_goal < 0.2 and self._collected_checkpoints[rew_index] < self._num_checkpoints:
                # Give checkpoint reward and update the collection
                components["checkpoint_reward"][rew_index] = self._checkpoint_reward
                reward[rew_index] += self._checkpoint_reward
                self._collected_checkpoints[rew_index] += 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
