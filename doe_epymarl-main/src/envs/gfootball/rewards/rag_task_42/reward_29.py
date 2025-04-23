import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for controlling the midfield and transitioning 
    strategically from defense to offense and vice versa.
    """
    def __init__(self, env):
        super().__init__(env)
        self.midfield_control_reward = 0.1
        self.transition_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        # Break down reward components for details
        components = {'base_score_reward': reward.copy(),
                      'midfield_control_reward': [0.0] * len(reward),
                      'transition_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for agent_idx, o in enumerate(observation):
            # Reward for controlling the midfield (around y-coordinates near zero)
            midfield_position_threshold = 0.1
            if abs(o['ball'][1]) < midfield_position_threshold:
                components['midfield_control_reward'][agent_idx] = self.midfield_control_reward
                reward[agent_idx] += components['midfield_control_reward'][agent_idx]
            
            # Reward for transitioning from defense to offense or vice versa
            previous_mode = components.get('previous_game_mode', None)
            if previous_mode and previous_mode != o['game_mode']:
                if o['game_mode'] == 1:  # Assuming '1' represents some sort of transition phase
                    components['transition_reward'][agent_idx] = self.transition_reward
                    reward[agent_idx] += components['transition_reward'][agent_idx]
            components['previous_game_mode'] = o['game_mode']

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
                if action:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
