import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes energy conservation in player movement and action choices."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros((2, 10), dtype=int)
        self.stamina_preservation_reward = 0.1
        self.prev_sticky_actions = np.zeros((2, 10), dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros((2, 10), dtype=int)
        self.prev_sticky_actions = np.zeros((2, 10), dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'prev_sticky_actions': self.prev_sticky_actions
        }
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = wrapper_state['sticky_actions_counter']
        self.prev_sticky_actions = wrapper_state['prev_sticky_actions']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            components[f"agent_{i}_stamina_preservation"] = 0
            current_actions = observation[i]['sticky_actions']
            # Reward reduction of movement actions compared to previous step
            movement_actions_indices = [0, 1, 2, 3, 4, 5, 6, 7] # Indices for movement actions
            total_movements_current = np.sum(current_actions[movement_actions_indices])
            total_movements_prev = np.sum(self.prev_sticky_actions[i][movement_actions_indices])
            
            if total_movements_current < total_movements_prev:
                reward_change = (total_movements_prev - total_movements_current) * self.stamina_preservation_reward
                reward[i] += reward_change
                components[f"agent_{i}_stamina_preservation"] = reward_change
            
            # Update previous actions
            self.prev_sticky_actions[i] = current_actions.copy()
            
        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value) if isinstance(value, list) else value
        return observation, reward, done, info
