import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies rewards for a football agent focusing on acting as a sweeper.
    It promotes clearing the ball from defensive zones, performing critical tackles, and fast recoveries.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.clear_distance_threshold = 0.5  # ball distance threshold to consider clearing successful
        self.tackle_success_threshold = 0.2  # tackle distance threshold
        self.recovery_speed_threshold = 0.02  # speed threshold to consider as a fast recovery
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)
    
    def reward(self, reward):
        # Extract useful observations for sweepers
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy()}
        
        if observation is None:
            return reward, components
        
        for idx, obs in enumerate(observation):
            self.modify_sweeper_rewards(obs, reward, idx, components)

        return reward, components

    def modify_sweeper_rewards(self, obs, reward, idx, components):
        # Check if the player is near the ball and the ball is in the defensive half
        if obs['ball'][0] < 0 and self.distance(obs['left_team'][obs['active']], obs['ball'][:2]) <= self.tackle_success_threshold:
            components.setdefault('tackle_reward', []).append(0.1)
            reward[idx] += 0.1  # Reward tackles in defensive half

        # Encourage playing the ball away from the goal if it's in the danger zone
        if obs['ball'][0] < -0.5:
            components.setdefault('clearing_reward', []).append(0.3)
            reward[idx] += 0.3  # High reward for clearing the ball from very defensive positions

        # Reward quick movements to position (recovery speed)
        if obs['left_team_direction'][obs['active']][1] > self.recovery_speed_threshold:
            components.setdefault('recovery_speed_reward', []).append(0.05)
            reward[idx] += 0.05  # Small reward for quick lateral movements

    def distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action_state
        return observation, reward, done, info
