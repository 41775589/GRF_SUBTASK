import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focused on incentivizing speed and precision with sprints and high passes on flanks."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._sprint_reward = 0.2 
        self._pass_reward = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sprint_actions_counter': self.sticky_actions_counter.copy()
        }
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', {}).get(
            'sprint_actions_counter', np.zeros(10, dtype=int))
        return from_pickle
    
    def reward(self, reward):
        components = {
            "base_score_reward": reward.copy(),
            "sprint_reward": [0.0],
            "pass_reward": [0.0]
        }
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            agent_obs = observation[rew_index]
            
            # Check if the agent is sprinting
            if agent_obs['sticky_actions'][8] == 1:  # action_sprint index is 8
                components['sprint_reward'][rew_index] = self._sprint_reward
                
            # Check if a high pass is performed
            if agent_obs['game_mode'] == 4:  # Assuming game mode 4 correlates to performing a high pass
                components['pass_reward'][rew_index] = self._pass_reward

        # Reward calculation
        for rew_index in range(len(reward)):
            reward[rew_index] += components['sprint_reward'][rew_index] + components['pass_reward'][rew_index]
        
        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
