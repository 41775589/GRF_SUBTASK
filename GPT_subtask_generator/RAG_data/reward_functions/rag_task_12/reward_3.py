import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for progressing in offensive and defensive tasks, focusing 
    on hybrid actions like high/long passing, dribbling under pressure, and sprint management."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "possession_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward),
            "pressing_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            base_reward = reward[rew_index]
            # Modifying reward based on possession under pressure
            if obs['ball_owned_team'] == 0 and obs['sticky_actions'][9]:  # Team 0 and dribbling
                components["possession_reward"][rew_index] = 0.1

            # Reward for successful long or high passes
            if ('action' in obs and obs['action'] in [football_action_set.action_long_pass, football_action_set.action_high_pass]):
                components["passing_reward"][rew_index] = 0.2

            # Reward for pressing effectively, which in this case means initiating sprints near opponents
            if obs['sticky_actions'][8] and min([np.linalg.norm(obs['right_team'][i] - obs['ball']) for i in range(len(obs['right_team']))]) < 0.1:
                components["pressing_reward"][rew_index] = 0.15

            # Sum the component rewards with the base rewards
            reward[rew_index] = base_reward + components["possession_reward"][rew_index] + components["passing_reward"][rew_index] + components["pressing_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
