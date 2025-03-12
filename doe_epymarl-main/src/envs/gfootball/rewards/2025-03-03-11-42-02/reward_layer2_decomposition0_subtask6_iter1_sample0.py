import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering defensive sliding tackles under pressure."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Constants for sliding tackle rewards
        self.sliding_tackle_reward = 1.0
        self.high_pressure_threshold = 0.2  # arbitrary threshold for closeness of opponents

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "sliding_tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the active player has performed a sliding tackle
            is_sliding = (o['sticky_actions'][6] == 1)  # Assume index 6 is sliding tackle

            # Check pressure by opponents
            dist_to_opponents = np.linalg.norm(o['right_team'] - o['left_team'][o['active']], axis=1)
            high_pressure = np.any(dist_to_opponents < self.high_pressure_threshold)

            # Increase the reward if a sliding tackle is made under high pressure
            if is_sliding and high_pressure:
                components["sliding_tackle_reward"][rew_index] = self.sliding_tackle_reward
                reward[rew_index] += self.sliding_tackle_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
