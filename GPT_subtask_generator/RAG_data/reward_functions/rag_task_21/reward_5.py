import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing defensive skills, particularly in responsiveness and interception."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_positions = np.linspace(-1, 1, 10)
        self._intercept_reward = 0.05
        self._block_reward = 0.05
        
    def reset(self):
        observation = self.env.reset()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return observation
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return to_pickle
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for blocking opponent close to our goal
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Opponent has the ball
                # Check the location of the ball relative to defense positions
                for dp in self._defensive_positions:
                    if np.abs(o['ball'][0] - dp) < 0.1:
                        components["defensive_reward"][rew_index] += self._block_reward
                        reward[rew_index] += components["defensive_reward"][rew_index]
            
            # Reward for intercepting the ball
            if 'ball_owned_team' not in o or o['ball_owned_team'] != 1:  # Our team intercepts the ball
                components["defensive_reward"][rew_index] += self._intercept_reward
                reward[rew_index] += components["defensive_reward"][rew_index]
            
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
