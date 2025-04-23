import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on defensive tackles without fouling."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.non_fouling_tackles_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, from_state):
        from_pickle = self.env.set_state(from_state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observations = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward)
        }
        
        if observations is None:
            return reward, components
        
        for i, o in enumerate(observations):
            # Reward for successful tackles without fouling
            tackle_actions = np.array([6, 7])  # indices for sliding and standing tackle actions
            fouls = o.get('yellow_card', np.array([False] * len(o['left_team'])))
            tackles = o.get('sticky_actions', [0] * len(tackle_actions))
            
            # Count tackles that did not result in fouls
            successful_tackles = sum(tackles[action_index] * (not fouls[player]) 
                                     for action_index, player 
                                     in enumerate(np.where(tackles > 0)[0]))
            
            # Apply reward for non-fouling tackles
            components["tackle_reward"][i] = self.non_fouling_tackles_reward * successful_tackles
            reward[i] += components["tackle_reward"][i]

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
