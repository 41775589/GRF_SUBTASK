import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful sliding tackles in defensive scenarios near the own goal."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sliding_tackle_reward = 0.5  # Reward given for a successful sliding tackle
        self.defensive_third = -0.33  # Define defensive third boundary on the x-axis near own goal
        self.tackle_success_counter = 0
        self.reset()

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "tackle_success_counter": self.tackle_success_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.tackle_success_counter = from_pickle['CheckpointRewardWrapper']['tackle_success_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_x = o['ball'][0]

            # Check if the ball is in the defensive third near own goal and if the controlled player is performing a sliding tackle
            if ball_x <= self.defensive_third and o['sticky_actions'][9] == 1 and o['ball_owned_team'] == 0:
                components["tackle_reward"][rew_index] = self.sliding_tackle_reward
                reward[rew_index] += components["tackle_reward"][rew_index]
                self.tackle_success_counter += 1

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
            for i, action_value in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_value
        info['tackle_successes'] = self.tackle_success_counter
        return observation, reward, done, info
