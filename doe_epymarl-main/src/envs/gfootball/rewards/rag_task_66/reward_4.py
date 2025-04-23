import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passes_completed = 0
        self.pass_reward = 0.5  # Reward for a successful pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passes_completed = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_score_reward = reward.copy()
        
        reward_components = {"base_score_reward": base_score_reward,
                             "pass_reward": [0.0] * len(reward)}
        
        if not observation:
            return reward, reward_components
        
        for idx, obs in enumerate(observation):
            # Reward for successful passes under pressure:
            if obs['game_mode'] in (3, 5):  # free kick or throw in
                if obs['ball_owned_team'] == 0 and obs['active'] == obs['ball_owned_player']:  # Our team has the ball
                    self.passes_completed += 1
                    reward_components["pass_reward"][idx] = self.pass_reward
                    reward[idx] += reward_components["pass_reward"][idx]
        
        return reward, reward_components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, reward_components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return obs, reward, done, info
