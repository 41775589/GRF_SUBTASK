import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds custom reward adjustments focusing on defensive skills."""

    def __init__(self, env):
        super().__init__(env)
        # Reward coefficients
        self.slide_tackle_reward = 0.5
        self.interception_reward = 0.3
        self.safe_pass_reward = 0.2
        self.positioning_reward = 0.1

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        modified_rewards = reward.copy()  # shallow copy of the original reward list
        components = {
            "base_score_reward": reward.copy(),
            "slide_tackle_reward": [0.0] * len(reward),
            "interception_reward": [0.0] * len(reward),
            "safe_pass_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o["team_leads"]:
                modified_rewards[rew_index] += self.positioning_reward
                components["positioning_reward"][rew_index] = self.positioning_reward

            if o["made_safe_pass"]:
                components["safe_pass_reward"][rew_index] = self.safe_pass_reward
                modified_rewards[rew_index] += self.safe_pass_reward

            if o["made_slide_tackle"]:
                components["slide_tackle_reward"][rew_index] = self.slide_tackle_reward
                modified_rewards[rew_index] += self.slide_tackle_reward

            if o["completed_interception"]:
                components["interception_reward"][rew_index] = self.interception_reward
                modified_rewards[rew_index] += self.interception_reward
            
        return modified_rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
