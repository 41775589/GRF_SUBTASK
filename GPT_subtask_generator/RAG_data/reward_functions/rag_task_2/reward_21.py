import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on defensive strategies and teamwork."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_balanced_defense = 0.0  # To track defensive balance rewards across steps

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_rewards": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            # Calculate the boosted defensive balance reward
            balance_reward = np.abs(o['left_team'][o['active']][0])
            # Increase reward based on how far back the active agent is toward their own goal
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                # If the left team owns the ball and it's being controlled by the active player
                components["defensive_rewards"][rew_index] += balance_reward

            # Update reward considering the defensive balance
            reward[rew_index] += components["defensive_rewards"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] = action

        return observation, reward, done, info
