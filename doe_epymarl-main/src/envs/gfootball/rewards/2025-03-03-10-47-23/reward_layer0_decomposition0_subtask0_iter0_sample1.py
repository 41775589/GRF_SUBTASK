import gym
import numpy as np
class CheckpointRewardWrapper(RewardWrapper):
    """Add subtask-specific checkpoint rewards for learning offensive skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._collected_checkpoints = {}
        self._num_checkpoints = 5
        self._checkpoint_reward = 0.1

    def reset(self):
        self._collected_checkpoints = {}
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "checkpoint_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball' in o and o['ball_owned_team'] == 0:
                d = ((o['ball'][0] - 1)**2 + o['ball'][1]**2)**0.5
                if d < 0.2:
                    components["checkpoint_reward"][rew_index] = self._checkpoint_reward * (
                            self._num_checkpoints - self._collected_checkpoints.get(rew_index, 0))
                    reward[rew_index] += components["checkpoint_reward"][rew_index]
                    self._collected_checkpoints[rew_index] = self._num_checkpoints

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
