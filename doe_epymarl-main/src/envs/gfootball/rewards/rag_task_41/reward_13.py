import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on offensive skills."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)
    
    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_play_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for index, o in enumerate(observation):
            components["offensive_play_reward"][index] = self.calculate_offensive_reward(o)
            reward[index] += components["offensive_play_reward"][index]
        
        return reward, components

    def calculate_offensive_reward(self, obs):
        """ Calculate rewards based on offensive play criteria. """
        offensive_reward = 0.0
        
        if obs['ball_owned_team'] == 0:  # If the left team (controlled team) owns the ball
            ball_position = obs['ball'][0]  # Get x position of the ball
            if ball_position > 0.5:  # Ball is on the opponent's half
                offensive_reward += 0.01
            
            if obs['active'] == obs['ball_owned_player']:  # If active player has the ball
                if obs['sticky_actions'][-1] == 1:  # Check if dribbling
                    offensive_reward += 0.02
                
        return offensive_reward
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += active
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
                
        return observation, reward, done, info
