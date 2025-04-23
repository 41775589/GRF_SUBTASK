import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A Gym wrapper that augments the reward function to emphasize learning specific offensive skills.
    This includes actions such as short passes, long passes, shots, dribbling, and sprinting to create scoring opportunities.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward coefficients
        self.short_pass_coefficient = 0.1
        self.long_pass_coefficient = 0.1
        self.shot_coefficient = 0.3
        self.dribble_coefficient = 0.2
        self.sprint_coefficient = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "short_pass_reward": [0.0] * len(reward),
            "long_pass_reward": [0.0] * len(reward),
            "shot_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "sprint_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sticky_actions = o['sticky_actions']
            if sticky_actions[7]:  # Short Pass action
                components["short_pass_reward"][rew_index] = self.short_pass_coefficient
            if sticky_actions[8]:  # Long Pass action
                components["long_pass_reward"][rew_index] = self.long_pass_coefficient
            if sticky_actions[0]:  # Shot action
                components["shot_reward"][rew_index] = self.shot_coefficient
            if sticky_actions[9]:  # Dribble action
                components["dribble_reward"][rew_index] = self.dribble_coefficient
            if sticky_actions[6]:  # Sprint action
                components["sprint_reward"][rew_index] = self.sprint_coefficient

            # Update reward based on the components
            reward[rew_index] += sum([
                components["short_pass_reward"][rew_index],
                components["long_pass_reward"][rew_index],
                components["shot_reward"][rew_index],
                components["dribble_reward"][rew_index],
                components["sprint_reward"][rew_index]
            ])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
