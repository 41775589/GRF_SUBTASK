import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a customized reward for training a goalkeeper."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the reward wrapper state and environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get current internal wrapper state."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set current internal wrapper state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Custom reward function designed for the goalkeeper."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_rewards": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            ball_position = obs['ball']
            ball_owned_team = obs['ball_owned_team']
            goalie_index = obs['left_team_roles'].index(0)  # Assumes goalie is role '0'
            
            # Punish for goals conceded
            if ball_position[0] > 0.9 and ball_owned_team == 1:  # Ball in near own goal
                components['goalkeeper_rewards'][i] -= 1.0
             
            # Reward for saves or clearing the ball from close to the goal
            if ball_position[0] < -0.75 and ball_owned_team == -1:  # Ball in dangerous area, not owned
                components['goalkeeper_rewards'][i] += 0.5
            
            # Distribution under pressure
            if obs['ball_owned_player'] == goalie_index and any(obs['sticky_actions'][1:4]):  # Goalkeeper has the ball and plays it
                components['goalkeeper_rewards'][i] += 0.3
            
            # Effective communication and positioning penalizing unnecessary movements
            if obs['ball_owned_team'] == 0:  # When ball is with the left team
                components['goalkeeper_rewards'][i] += (1 - np.abs(ball_position[1])) * 0.1  # Encourage central positioning
            
            # Apply the components to the computed reward
            reward[i] += components['goalkeeper_rewards'][i]
        
        return reward, components

    def step(self, action):
        """Execute an action and observe the outcome."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
