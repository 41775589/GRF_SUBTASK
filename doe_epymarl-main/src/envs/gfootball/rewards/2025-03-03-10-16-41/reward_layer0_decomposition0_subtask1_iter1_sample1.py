import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a checkpoint reward for defensive actions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._collected_checkpoints = {}
        self._num_checkpoints = 5
        self._checkpoint_reward = 0.1

    def reset(self):
        self._collected_checkpoints = {}
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if reward[rew_index] == -1:
                components["checkpoint_reward"][rew_index] = self._checkpoint_reward * (
                        self._num_checkpoints -
                        self._collected_checkpoints.get(rew_index, 0))
                reward[rew_index] = -1 * components["base_score_reward"][rew_index] + components["checkpoint_reward"][
                    rew_index]
                self._collected_checkpoints[rew_index] = self._num_checkpoints
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
