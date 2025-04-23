import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on defensive responsiveness and interception skills."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # initializes counter of sticky actions
        
    def reset(self):
        """Resets the environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Get the internal state including the base environment state."""
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        """Set the internal state including the base environment state."""
        return self.env.set_state(state)
    
    def reward(self, reward):
        """Custom reward function to enhance defensive positioning and interception capabilities."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'defensive_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index, _ in enumerate(reward):
            o = observation[rew_index]
            
            # Encourage maintaining close proximity to the opponent holding the ball and intercept passes
            if o['ball_owned_team'] == 1:  # If opponent has the ball
                ball_position = np.array(o['ball'][:2])  # ignore z-coordinate
                player_position = np.array(o['left_team'][o['active']])
                distance_to_ball = np.linalg.norm(ball_position - player_position)
                
                # Reward for being close to the ball when the opponent team owns it
                if distance_to_ball < 0.1:
                    components['defensive_reward'][rew_index] = 0.5
            
            # Encourage gaining possession from the opponent
            prev_ball_owned_team = self.env.unwrapped.get_state({}).get('prev_ball_owned_team', -1)
            if prev_ball_owned_team == 1 and o['ball_owned_team'] == 0:
                components['defensive_reward'][rew_index] += 1.0  # High reward for gaining possession
            
            # Calculate the final modified reward by adding the additional component
            reward[rew_index] += components['defensive_reward'][rew_index]
        
        return reward, components
    
    def step(self, action):
        """Step function to perform action, retrieve next state and modified reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
