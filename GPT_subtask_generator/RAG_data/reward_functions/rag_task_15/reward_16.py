import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a specific reward for practicing long passes across various field sections.
    It provides incremental rewards for successfully passing the ball over longer distances.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_thresholds = np.linspace(0.2, 1.0, 5)  # Define different distances as checkpoints for long passes
        self.reward_for_pass = 0.05  # Reward given for each successful pass crossing a threshold
        self.last_ball_position = None
        self.pass_reached = np.zeros_like(self.pass_thresholds, dtype=bool)

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.pass_reached = np.zeros_like(self.pass_thresholds, dtype=bool)
        return self.env.reset(**kwargs)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "long_pass_reward": 0.0}
        current_ball_position = observation['ball'][:2]  # Ignore the z coordinate
        
        if self.last_ball_position is not None:
            distance = np.linalg.norm(current_ball_position - self.last_ball_position)
            cumulative_distance = np.cumsum(self.pass_thresholds >= distance)
            new_passes = self.pass_thresholds[~self.pass_reached & (cumulative_distance > 0)]
            incremental_reward = len(new_passes) * self.reward_for_pass
            components["long_pass_reward"] = incremental_reward
            reward += incremental_reward
            self.pass_reached |= cumulative_distance > 0
        
        self.last_ball_position = current_ball_position  # Update the last known position
        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = reward

        # Log reward components
        for key, value in components.items():
            info[f"component_{key}"] = value
        
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_idx, agent_obs in enumerate(obs):
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
