import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for optimizing shooting angles and timing near the goal.
    This is designed for finishing under high-pressure scenarios close to the opponent's goal.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_zone_threshold = 0.2  # Define how close to the goal line (x = 1) to consider shooting zone
        self.angle_reward_coefficient = 0.5  # Coefficient for calculating the angle reward part
        self.timing_reward_coefficient = 0.5  # Coefficient for timing reward

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
        components = {"base_score_reward": reward.copy()}

        for rew_index, o in enumerate(observation):
            # Default to initialization of additional reward components
            angle_reward = 0
            timing_reward = 0

            # Calculate new rewards components based on ball position and player possession
            if o['ball_owned_team'] == 1 and o['ball'][0] > (1 - self.shooting_zone_threshold):
                # If ball is near the opponent's goal, calculate angle optimization reward
                angle = abs(np.arctan2(o['ball'][1], o['ball'][0] - 1))  # Angle with the vertical line of the goal
                angle_reward = self.angle_reward_coefficient * (np.pi/2 - angle) / (np.pi/2)
            
                # Timing reward would be considered based on game steps left or game modes, not exemplified here
                timing_reward = self.timing_reward_coefficient * (self.env.unwrapped.steps_left / self.env.unwrapped.initial_steps)
                
            # Aggregate the shaped rewards to the original rewards array
            total_shaped_reward = angle_reward + timing_reward
            reward[rew_index] += total_shaped_reward
            components[f'angle_reward_{rew_index}'] = angle_reward
            components[f'timing_reward_{rew_index}'] = timing_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        final_reward = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        info["final_reward"] = final_reward
        obs = self.env.unwrapped.observation()
        
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action

        return observation, reward, done, info
