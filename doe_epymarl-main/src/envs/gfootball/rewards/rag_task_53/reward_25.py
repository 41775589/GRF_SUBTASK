import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for maintaining ball control and effective field transitions to exploit open spaces."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Create specific zones within the field to encourage strategic plays across different areas
        self.zones = [
            (-1.0, -0.33), (-1.0, 0.33), (-0.66, -0.33), (-0.66, 0.33),
            (0.0, -0.33), (0.0, 0.33), (0.33, -0.33), (0.33, 0.33),
            (1.0, -0.33), (1.0, 0.33)
        ]
        self.zone_rewards = np.zeros(len(self.zones), dtype=float)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.zone_rewards = np.zeros(len(self.zones), dtype=float)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'zone_rewards': self.zone_rewards
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        res = self.env.set_state(state)
        from_pickle = res['CheckpointRewardWrapper']
        self.zone_rewards = from_pickle['zone_rewards']
        return res

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_reward": [0.0] * len(reward),
                      "strategic_transition_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            ball_position = o['ball'][:2]
            ball_control = o['ball_owned_team'] == 0

            # Reward for maintaining possession of the ball and avoiding losing it under pressure
            if ball_control:
                components["possession_reward"][rew_index] += 0.02

            # Reward for strategically transitioning the ball across different field zones effectively
            for i, (x, y) in enumerate(self.zones):
                if (ball_position[0] > x - 0.33 and ball_position[0] < x + 0.33) and \
                   (ball_position[1] > y - 0.42 and ball_position[1] < y + 0.42):
                    if self.zone_rewards[i] == 0: # Zone is freshly controlled
                        components["strategic_transition_reward"][rew_index] += 0.05
                        self.zone_rewards[i] = 1  # Mark zone as rewarded

            # Calculate the final wrapped reward for the current observation
            reward_sum = components["base_score_reward"][rew_index] + \
                         components["possession_reward"][rew_index] + \
                         components["strategic_transition_reward"][rew_index]
            reward[rew_index] = reward_sum

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        # Add individual components to info for monitoring
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        # Update sticky actions counter based on observations
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action  # record in info which actions are being held
        return observation, reward, done, info
