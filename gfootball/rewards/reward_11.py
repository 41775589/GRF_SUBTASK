import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for offensive strategies like shooting, dribbling, and passing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initializing counters and other necessary variables
        self.shooting_reward = 0.3
        self.dribble_reward = 0.2
        self.passing_reward = 0.1

    def reset(self):
        """Reset the environment and any additional components."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store the current state of the wrapper along with the environment's state."""
        to_pickle['CustomRewardWrapper'] = {} # You might store specifics if needed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state from a saved state."""
        from_pickle = self.env.set_state(state)
        # Restore any specific state if needed
        # Example: self.my_state = from_pickle['CustomRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Rewards are modified based on the environment's behavior."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 
                      'shooting_reward': [0.0], 
                      'dribbling_reward': [0.0], 
                      'passing_reward': [0.0]}

        if observation is None:
            return reward, components

        for o in observation:
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:  # Checks if the ball is owned by the controlled team
                if 'action' in o:
                    action_type = o['action']
                    if action_type == 'shot':
                        reward += self.shooting_reward
                        components['shooting_reward'][0] = self.shooting_reward
                    elif action_type == 'dribble':
                        reward += self.dribble_reward
                        components['dribbling_reward'][0] = self.dribble_reward
                    elif action_type == 'long_pass' or action_type == 'high_pass':
                        reward += self.passing_reward
                        components['passing_reward'][0] = self.passing_reward

        return reward, components

    def step(self, action):
        """Step function to execute environment step and modify reward."""
        observation, reward, done, info = self.env.step(action)
        
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        
        # Add final reward to the info so it can be easily accessed if needed
        info['final_reward'] = reward
        
        # Add individual components to info for detailed analysis
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        
        return observation, reward, done, info
