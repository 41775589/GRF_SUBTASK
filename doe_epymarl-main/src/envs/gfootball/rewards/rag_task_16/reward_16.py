import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for executing precision high passes."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_effectiveness_coefficient = 2.0
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "high_pass_skill_enhancement": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]
            
            if 'ball_owned_team' not in o or o['ball_owned_team'] != o['active']:
                continue
            
            # Check if a high pass is executed (sprint + direction upwards)
            if o['sticky_actions'][8] == 1 and (o['sticky_actions'][2] == 1 or o['sticky_actions'][3] == 1):
                # The player is performing a high pass, evaluate its effectiveness based on trajectories
                ball_direction = o['ball_direction']
                # Assuming effective high pass is when ball z-direction (height) increases significantly
                if ball_direction[2] > 0.05:  # purely indicative threshold
                    components["high_pass_skill_enhancement"][rew_index] = self.high_pass_effectiveness_coefficient
                    reward[rew_index] += components["high_pass_skill_enhancement"][rew_index]
                    
        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action_bool in enumerate(agent_obs['sticky_actions']):
                if action_bool:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
                
        return observation, reward, done, info
