import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances defensive capabilities by adding specific rewards for 
    defensive actions such as successful tackles, intercepting the ball, and goalkeeping.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # No specific state needed for this reward wrapper
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        # Reward adjustments - more defensive adjustments can be added as necessary
        tackle_reward_coeff = 0.2
        intercept_reward_coeff = 0.2
        goalkeepers_ball_save_coeff = 0.5

        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": 0.0,
            "intercept_reward": 0.0,
            "goalkeepers_ball_save": 0.0
        }

        if observation is None:
            return reward, components

        for player_obs in observation:
            # Customize based on your game observation for tackling and intercepts
            # Dummy conditions for demo purposes, should be replaced with actual game logic
            if player_obs.get('tackle', False):
                components["tackle_reward"] += tackle_reward_coeff
                reward += tackle_reward_coeff
            if player_obs.get('intercept', False):
                components["intercept_reward"] += intercept_reward_coeff
                reward += intercept_reward_coeff
            if player_obs.get('role') == 'goalkeeper' and player_obs.get('ball_save', False):
                components["goalkeepers_ball_save"] += goalkeepers_ball_save_coeff
                reward += goalkeepers_ball_save_coeff

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
