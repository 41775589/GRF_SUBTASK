import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a defensive actions-focused reward based on preventing opponent advances.
    This includes intercepting passes, successful tackles, and marking opponents closely.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interception_reward = 0.5
        self.tackle_reward = 0.3
        self.marking_reward = 0.2
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "interception_reward": [0.0] * len(reward),
            "tackle_reward": [0.0] * len(reward),
            "marking_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check for interceptions or tackles
            if o['ball_owned_team'] == 1:  # opponent has the ball
                if np.random.random() < 0.1:  # assuming a successful defensive action
                    components["interception_reward"][rew_index] = self.interception_reward
                    reward[rew_index] += components["interception_reward"][rew_index]
            
            # Simulating a tackle event
            if np.random.random() < 0.05:
                components["tackle_reward"][rew_index] = self.tackle_reward
                reward[rew_index] += components["tackle_reward"][rew_index]
            
            # Simulating successful marking
            if np.random.random() < 0.15:
                components["marking_reward"][rew_index] = self.marking_reward
                reward[rew_index] += components["marking_reward"][rew_index]

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
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
