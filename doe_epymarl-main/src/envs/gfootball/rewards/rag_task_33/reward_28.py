import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides a reward for successful long shots outside the penalty box, 
       encouraging training on long-range shooting techniques."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.previous_ball_position = np.array([0.0, 0.0])
        # Starting shot-long-range shots from beyond 0.6 x-coordinate in absolute values is considered long-range
        self.long_range_limit = 0.6
        self.penalty_box_x = 0.8  # x-coordinate range considered as inside penalty box
        self.reward_for_long_shot = 1.0  # Reward multiplier for successful long shot
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_score = [0, 0]

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_score = [0, 0]
        observation = self.env.reset()
        self.previous_ball_position = observation['ball'][:2]
        return observation

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        current_ball_position = observation['ball'][:2]
        score_change = np.array(observation['score']) - np.array(self.previous_score)
        reward_components = {"base_score_reward": float(sum(reward))}
        
        if abs(current_ball_position[0]) > self.long_range_limit and \
           abs(self.previous_ball_position[0]) > self.long_range_limit and \
           abs(current_ball_position[0]) < self.penalty_box_x:
            if score_change[0] > 0 or score_change[1] > 0:  # Assuming scoring for left team is positive reward
                additional_reward = self.reward_for_long_shot
                reward_components['long_shot_reward'] = additional_reward
                reward[0] += additional_reward  # Modify the reward for the left agent if goal is scored  

        # Save the current state for the next step
        self.previous_score = observation['score']
        self.previous_ball_position = current_ball_position

        return reward, reward_components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
