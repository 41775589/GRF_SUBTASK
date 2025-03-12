import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for offensive group training."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._checkpoint_reward = 0.1
        self._checkpoints_collected = {agent_id: 0 for agent_id in range(3)}
        self._num_checkpoints = 10

    def reset(self):
        self._checkpoints_collected = {agent_id: 0 for agent_id in range(3)}
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "checkpoint_reward": [0.0] * len(reward)}

        for agent_id in range(3):
            agent_observation = observation[agent_id]
            checkpoint_distance = ((agent_observation['ball'][0] - 1) ** 2 + agent_observation['ball'][1] ** 2) ** 0.5

            if checkpoint_distance <= 0.2 and self._checkpoints_collected[agent_id] < self._num_checkpoints:
                components["checkpoint_reward"][agent_id] = self._checkpoint_reward
                self._checkpoints_collected[agent_id] += 1
                reward[agent_id] += self._checkpoint_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
