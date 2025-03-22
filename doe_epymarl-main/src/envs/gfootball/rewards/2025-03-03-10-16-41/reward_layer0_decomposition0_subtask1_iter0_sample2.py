import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._collected_checkpoints = {}
        self._checkpoint_reward = 0.1
        self._num_checkpoints = 5

    def reset(self):
        self._collected_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if ('ball_owned_team' not in o or
                    o['ball_owned_team'] == 1):
                d = ((o['ball'][0] + 1) ** 2 + o['ball'][1] ** 2) ** 0.5
                if (self._collected_checkpoints.get(rew_index, 0) < self._num_checkpoints) and d < 0.8:
                    components["checkpoint_reward"][rew_index] = self._checkpoint_reward
                    reward[rew_index] += 1.5 * components["checkpoint_reward"][rew_index]
                    self._collected_checkpoints[rew_index] = (
                            self._collected_checkpoints.get(rew_index, 0) + 1)
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
