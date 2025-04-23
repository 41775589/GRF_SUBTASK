import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on high passes and crossing strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_reward = 0.5

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
        new_rewards = []
        components = {"base_score_reward": [], "high_pass_bonus": []}
        observation = self.env.unwrapped.observation()
        
        for i, obs in enumerate(observation):
            components["base_score_reward"].append(reward[i])
            high_pass_bonus = 0
            
            # Encourage high and long passes: high pass actions and specific game modes
            if obs['ball_direction'][2] > 0.1 + 0.05*np.random.rand():  # Simulating variability in pass quality
                # Additional reward for executing a high ball
                high_pass_bonus = self.high_pass_reward
                components["high_pass_bonus"].append(high_pass_bonus)
            else:
                components["high_pass_bonus"].append(0)

            new_rewards.append(reward[i] + high_pass_bonus)

        return new_rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
