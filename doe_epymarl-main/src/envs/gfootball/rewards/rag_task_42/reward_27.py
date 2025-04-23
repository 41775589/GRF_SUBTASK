import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a midfield dynamics mastery reward, focusing on enhanced coordination,
    and strategic repositioning during offensive and defensive transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.midfield_control_rewards = {}
        self.ball_position_threshold = 0.2  # Threshold to consider midfield
        self.reward_for_midfield_control = 0.5

    def reset(self):
        self.midfield_control_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.midfield_control_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_control_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            if 'ball' in o:
                ball_x_pos = o['ball'][0]
                # Check if the ball is in the midfield region
                if -self.ball_position_threshold < ball_x_pos < self.ball_position_threshold:
                    # Reward the active player for controlling the midfield
                    components["midfield_control_reward"][rew_index] = self.reward_for_midfield_control
                    reward[rew_index] += components["midfield_control_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
