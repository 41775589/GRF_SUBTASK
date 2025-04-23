import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for mastering long passes with precision.
    It evaluates the accuracy and technical execution of long passes under varying match conditions.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Coefficient values for different pass lengths
        self.pass_accuracy_reward_coeff = 0.1
        self.long_pass_threshold = 0.3  # Minimum distance to consider a pass as long

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "pass_accuracy_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Calculate the distance the ball was passed by comparing positions from last observation
            if o['ball_owned_team'] == 0:  # Only consider the team we are training
                if abs(o['ball'][0]) - abs(o['ball_direction'][0] + o['ball'][0]) > self.long_pass_threshold:
                    distance_traveled = np.linalg.norm(o['ball_direction'][:2])
                    if distance_traveled > self.long_pass_threshold:
                        accuracy = max(0, 1 - np.abs(o['ball_direction'][1] / o['ball_direction'][0]))
                        pass_reward = accuracy * self.pass_accuracy_reward_coeff
                        components["pass_accuracy_reward"][rew_index] = pass_reward
                        reward[rew_index] += pass_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
