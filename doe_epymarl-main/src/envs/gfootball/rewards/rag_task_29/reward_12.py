import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper for enhancing shot precision skills specifically for scenarios within close range of the goal."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Define key regions for precision training near the goal
        self.goal_distance_thresholds = np.linspace(0.2, 0.02, 10)  # Close to goal dense thresholds
        self.close_range_shot_reward = 0.5  # Extra reward for shooting from these distances
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
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
        components = {"base_score_reward": reward.copy(),
                      "precision_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            ball_pos = o.get('ball', [])
            own_goal_pos = [-1, 0]  # x-position is negative for left player's goal
            target_goal_pos = [1, 0]  # x-position is positive for the right opponent's goal
            
            # Calculate distance to opponent's goal (x-axis normalization from -1 to 1)
            distance_to_goal = np.abs(target_goal_pos[0] - ball_pos[0])

            # Check if the distance to goal is within the specified close-range thresholds
            if distance_to_goal <= self.goal_distance_thresholds[0]:
                components["precision_reward"][rew_index] = self.close_range_shot_reward
                reward[rew_index] += components["precision_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
