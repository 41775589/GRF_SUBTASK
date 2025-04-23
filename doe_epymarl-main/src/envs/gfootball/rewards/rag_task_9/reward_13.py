import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward system focusing on offensive skills."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_reward = 0.1
        self.shooting_reward = 0.2
        self.dribbling_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        # Initialize reward components for each agent
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": [0.0] * len(reward),
            "shooting_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward)
        }
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            # Check for ball possession
            if o['ball_owned_team'] != 0:
                continue

            # Passing rewards (both short and long pass)
            if o['sticky_actions'][0] or o['sticky_actions'][1]:
                components["passing_reward"][rew_index] += self.passing_reward
                reward[rew_index] += self.passing_reward
            
            # Shooting reward
            if o['sticky_actions'][2]:
                components["shooting_reward"][rew_index] += self.shooting_reward
                reward[rew_index] += self.shooting_reward
            
            # Dribbling and sprinting rewards
            if o['sticky_actions'][9] and o['sticky_actions'][8]:
                components["dribbling_reward"][rew_index] += self.dribbling_reward
                reward[rew_index] += self.dribbling_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
