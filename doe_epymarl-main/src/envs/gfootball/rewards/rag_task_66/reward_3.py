import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to encourage mastering the technique of short passing under defensive pressure,
    focusing on ball retention and effective distribution.
    """
    def __init__(self, env):
        super().__init__(env)
        self.passing_threshold = 0.1  # Minimum change in ball direction to consider a pass.
        self.defensive_pressure_threshold = 0.2  # Euclidean distance to the nearest opponent.
        self.pass_reward = 0.5  # Reward for a successful pass.
        self.pressure_reward = 0.3  # Reward for maintaining ball under pressure.
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "pressure_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = 1 if np.linalg.norm(o['ball_direction']) > self.passing_threshold else 0
            nearest_opponent_distance = np.min(np.sqrt((o['right_team'] - o['ball'][:2])**2).sum(axis=1))
            if nearest_opponent_distance < self.defensive_pressure_threshold:
                components["pressure_reward"][rew_index] = self.pressure_reward
                reward[rew_index] += components["pressure_reward"][rew_index]

            if 'sticky_actions' in o and np.any(o['sticky_actions'][[8, 9]]):  # Actions related to dribbling or sprinting
                components["pass_reward"][rew_index] = self.pass_reward
                reward[rew_index] += components["pass_reward"][rew_index]

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
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
