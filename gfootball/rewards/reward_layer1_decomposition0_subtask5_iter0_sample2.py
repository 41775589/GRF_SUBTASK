import gym
import numpy as np


class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that adds a defensive-oriented reward system. """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Augment reward based on successful defensive actions."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            components["defensive_reward"][rew_index] = 0

            # Check if the game mode is a defensive situation and reward accordingly.
            if obs['game_mode'] in [3, 4, 5, 6]:  # FreeKick, Corner, ThrowIn, Penalty
                # Increase reward when successfully defending these situations
                if obs['ball_owned_team'] == 1:  # If the opponent team has the ball
                    defensive_boost = 0.5
                else:
                    defensive_boost = -0.2  # Penalize if you lose the ball
                components["defensive_reward"][rew_index] = defensive_boost
                reward[rew_index] += defensive_boost

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

