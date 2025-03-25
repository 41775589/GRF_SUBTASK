import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on executing high passes with precision."""

    def __init__(self, env):
        super().__init__(env)
        self.high_pass_activation_counter = np.zeros(10, dtype=int)  # Tracks activation of high pass actions

    def reset(self):
        self.high_pass_activation_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.high_pass_activation_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.high_pass_activation_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        # Initialize observations of the environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for executing a high pass
            if o['sticky_actions'][9] == 1:  # Assuming index 9 is the high pass action
                if self.high_pass_activation_counter[rew_index] == 0:
                    components["high_pass_reward"][rew_index] = 1.0  # Reward for first high pass
                    self.high_pass_activation_counter[rew_index] = 1
                reward[rew_index] += components["high_pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
