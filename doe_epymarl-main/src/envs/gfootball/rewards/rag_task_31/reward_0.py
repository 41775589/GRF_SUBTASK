import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effective tackling and aggressive defensive plays."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_reward = 0.5
        self.slide_reward = 0.8

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        pickle_data = self.env.get_state(to_pickle)
        pickle_data['sticky_actions_counter'] = self.sticky_actions_counter
        return pickle_data

    def set_state(self, state):
        state_data = self.env.set_state(state)
        self.sticky_actions_counter = state_data.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return state_data

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward),
            "slide_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            agent_obs = observation[rew_index]
            
            # Check for effective tackle action
            if agent_obs['sticky_actions'][9] == 1:  # Index 9 corresponds to the 'action_tackle'
                components["tackle_reward"][rew_index] = self.tackle_reward
                reward[rew_index] += self.tackle_reward
            
            # Check for slide action usage
            if agent_obs['sticky_actions'][8] == 1:  # Index 8 corresponds to the 'action_slide'
                components["slide_reward"][rew_index] = self.slide_reward
                reward[rew_index] += self.slide_reward
        
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
