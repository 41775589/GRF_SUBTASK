import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for ball control and effective dribbling with sprint actions."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Count of the sticky actions
        
        # This is the reward given for performing dribble while sprinting
        self.sprint_dribble_reward = 0.05
      
    def reset(self):
        """Reset the environment and reward parameters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
      
    def get_state(self, to_pickle):
        """Retrieve the state of the environment with added wrapper configuration."""
        to_pickle = self.env.get_state(to_pickle)
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return to_pickle
      
    def set_state(self, state):
        """Set the state of the environment from a pickled dictionary."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle
    
    def reward(self, reward):
        """Modify the reward based on ball control and sprint-dribble actions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_dribble_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for idx, o in enumerate(observation):
            # Check if the player has the ball control
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:  # Assuming 0 is the index for the controlled team
                # Check if 'sprint' and 'dribble' actions are active simultaneously
                if o['sticky_actions'][8] == 1 and o['sticky_actions'][9] == 1:  # 'sprint' at index 8, 'dribble' at 9
                    components["sprint_dribble_reward"][idx] = self.sprint_dribble_reward
                    reward[idx] += components["sprint_dribble_reward"][idx]
        
        return reward, components
    
    def step(self, action):
        """Step through the environment and modify the reward."""
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
