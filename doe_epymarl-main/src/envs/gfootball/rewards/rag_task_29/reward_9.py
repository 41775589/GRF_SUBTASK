import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for accurate shooting from close range. 
    The goal is to promote scoring by shooting from tight angles and positions close 
    to the enemy goal.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define zones near the goal and appropriate rewards
        self.close_range_zones = [(0.95, 1.0, 0.1), (0.9, 0.95, 0.05)]
        self.angle_accuracy_reward = 0.2
        self.power_accuracy_reward = 0.3

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
                      "close_range_shooting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Iterate through each agent's reward and observation
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['ball_owned_team'] == 1 and o['active'] == o['ball_owned_player']:
                # Calculate distance to the center of the opponent's goal
                distance_to_goal = abs(o['ball'][0] - 1.0)
                # Check close range zones
                for min_dist, max_dist, zone_reward in self.close_range_zones:
                    if min_dist <= distance_to_goal < max_dist:
                        components['close_range_shooting_reward'][rew_index] += zone_reward
                    
                    # Adjust for angle and power if a goal was scored
                    if reward[rew_index] == 1:
                        ball_direction = o['ball_direction']
                        ball_power = np.linalg.norm(ball_direction)
                        # Ideal shoot power for close range is high
                        if ball_power > 0.1:
                            reward[rew_index] += self.power_accuracy_reward * 0.5

                        # Ideal angle is towards the center of the goal (y close to 0)
                        if abs(ball_direction[1]) < 0.02:
                            reward[rew_index] += self.angle_accuracy_reward * 0.5
                            
            reward[rew_index] += components['close_range_shooting_reward'][rew_index]

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
            for i, action_p in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_p
        return observation, reward, done, info
