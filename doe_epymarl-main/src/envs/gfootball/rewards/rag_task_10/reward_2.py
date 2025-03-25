import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a defense-focused reward based on defensive actions such as 
    interception, marking, and tackling to prevent the opponent from scoring.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Sticky actions tracker

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward}
        
        # If observations are not available, return current reward
        if observation is None:
            return reward, components

        # Implementing reduction in reward for opponent's attack progress
        defense_reward = [0.0, 0.0]  # Two players in 2-agent scenario
        
        for index, obs in enumerate(observation):
            # Check if the opponent team has the ball
            if obs['ball_owned_team'] == 1:  # Assuming 1 is opponent and 0 is self team
                distance_to_goal = (obs['ball'][0] + 1)/2  # Transform field coords to [0, 1] range
                defense_reward[index] -= 0.1 * distance_to_goal  # Reduce reward based on proximity to goal
            
            # Encourage marking: check proximity to opponents
            player_pos = obs['right_team'][index] if obs['ball_owned_team'] == 0 else obs['left_team'][index]
            opponents = obs['left_team'] if obs['ball_owned_team'] == 0 else obs['right_team']
            close_opponents = np.sum(np.linalg.norm(player_pos - opponents, axis=1) < 0.1)
            defense_reward[index] += 0.05 * close_opponents  # Reward for being close to opponents (marking)
            
            # Handling specific sticky actions related to defense
            if (obs['ball_owned_team'] == 1) and (obs['sticky_actions'][7] or obs['sticky_actions'][0]):
                # If opponent has ball and defense actions like bottom_left (slide) or action_left (stop) are taken
                defense_reward[index] += 0.2
        
        reward = [r + d for r, d in zip(reward, defense_reward)]
        
        components['defense_reward'] = defense_reward  # Add defense components to the reward for monitoring
        return reward, components        
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Attach additional info related to reward components
        for key, value in components.items():
            info[f"component_{key}"] = value
        
        # Track sticky actions activation for debugging/analysis
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        
        return observation, reward, done, info
