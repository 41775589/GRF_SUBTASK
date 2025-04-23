import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing shot precision skills from close ranges and tight angles."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goal_proximity_reward = 0.2
        self.angle_precision_reward = 0.1
        self.power_adjustment_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goal_proximity_reward": [0.0] * len(reward),
                      "angle_precision_reward": [0.0] * len(reward),
                      "power_adjustment_reward": [0.0] * len(reward)}

        # Loop through each agent's reward and update accordingly
        for rew_index, rew in enumerate(reward):
            o = observation[rew_index]
            
            # Goal proximity reward
            distance_to_goal = abs(o['ball'][0] - 1)  # Normalize by field length
            if distance_to_goal < 0.2:  # Close range
                components["goal_proximity_reward"][rew_index] = self.goal_proximity_reward

            # Angle precision reward
            if o['ball_owned_team'] == 1:  # If the right team owns the ball
                if o['ball_owned_player'] == o['active']:
                    angle_to_goal = abs(o['ball'][1]/o['ball'][0])  # y/x for right side
                    if angle_to_goal < 0.1:  # Tight angle
                        components["angle_precision_reward"][rew_index] = self.angle_precision_reward

            # Power adjustment reward based on the previous action's effectiveness
            if self.sticky_actions_counter[9] == 1:  # Assuming index 9 is a powerful shot action
                components["power_adjustment_reward"][rew_index] = self.power_adjustment_reward
            
            # Combination of base reward and additional components
            additional_rewards = components["goal_proximity_reward"][rew_index] + \
                                 components["angle_precision_reward"][rew_index] + \
                                 components["power_adjustment_reward"][rew_index]
            reward[rew_index] += additional_rewards

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
