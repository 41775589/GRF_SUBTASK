import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for developing individual finishing skills."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._checkpoint_reward = 1.0
        self._num_checkpoints = 10

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        components = {"base_score_reward": reward.copy(), "checkpoint_reward": [0.0]}

        if observation is None:
            return reward, components

        o = observation[0]
        x, y = o['ball'][0], o['ball'][1]

        # Checking distance from the goal to simulate checkpoints
        distance_to_goal = ((x - 1) ** 2 + y ** 2) ** 0.5

        if distance_to_goal < 0.1:
            checkpoint_reward = self._checkpoint_reward * (self._num_checkpoints - 1)
            reward += 1.5 * checkpoint_reward
            components["checkpoint_reward"][0] = checkpoint_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
