import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides a dense, checkpoint-based reward for training offensive football strategies.
    It focuses on promoting effective dribbling, executing different types of passes, and mastering accurate shooting.
    """
    
    def __init__(self, env):
        super().__init__(env)
        # Checkpoints for dribbling skills enhancement
        self.dribble_threshold = 0.1  # Dribbling proximity to opponents to gain extra reward
        self.dribble_reward = 0.05
        
        # Passing checkpoints - long and high passes
        self.pass_reward = 0.1
        self.long_pass_threshold = 0.3  # Distance threshold for a long pass
        
        # Shooting accuracy checkpoints
        self.shooting_proximity_reward = 0.2
        self.close_to_goal_threshold = 0.1  # Proximity to goal for enhanced shooting rewards

    def reset(self):
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "pass_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned = o["ball_owned_team"] == 0
            
            # Enhancing dribbling rewards: checking the proximity to the opponents
            if ball_owned:
                player_pos = o['right_team'][o['active']]
                opponents = o['left_team']                
                distances = np.linalg.norm(opponents - player_pos, axis=1)
                if (distances < self.dribble_threshold).any():
                    components["dribble_reward"][rew_index] = self.dribble_reward
                    reward[rew_index] += components["dribble_reward"][rew_index]
                
            # Check for shooting rewards
            ball_pos = o['ball'][:2]
            goal_pos = [1, 0]  # Right team's attacking direction towards goal at x=1, y=0
            distance_to_goal = np.linalg.norm(ball_pos - goal_pos)
            if ball_owned and distance_to_goal < self.close_to_goal_threshold:
                components["shooting_reward"][rew_index] = self.shooting_proximity_reward
                reward[rew_index] += components["shooting_reward"][rew_index]
            
            # Rewards for different types of passes
            if ball_owned:
                if 'action' in o and o['action'] == 'long_pass' and distance_to_goal > self.long_pass_threshold:
                    components["pass_reward"][rew_index] = self.pass_reward
                    reward[rew_index] += components["pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
