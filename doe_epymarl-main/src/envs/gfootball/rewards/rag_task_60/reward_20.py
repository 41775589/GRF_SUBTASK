import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for reactive defensive transitions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state
    
    def set_state(self, state):
        return self.env.set_state(state)
    
    def reward(self, reward):
        # Extract environment observations
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_positioning_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            obs = observation[i]
            # Reward for returning to a tactical defensive position quickly after engagement
            if obs['game_mode'] in {2, 3, 4, 5, 6}:  # Involves modes related to set-pieces or interruptions
                if ('ball_owned_team' in obs and obs['ball_owned_team'] != obs['active'] and
                        (obs['active'] in obs['designated'])):
                    components['defensive_positioning_reward'][i] = 1.0  # Strong reward for transitioning quickly
                    
            # Modify reward based on components
            reward[i] = reward[i] + components['defensive_positioning_reward'][i]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
