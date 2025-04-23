import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for maintaining ball control under pressure and exploiting open spaces."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "ball_control_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Calculate ball control rewards
            ball_ownership = o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['designated']
            ball_position_relative_to_goal = 1.0 - np.abs(o['ball'][0])  # Closer to 0 is closer to opponent's goal

            if ball_ownership:
                components["ball_control_reward"][rew_index] += 0.1 * ball_position_relative_to_goal

            # Calculate positional play rewards
            if ball_ownership and ball_position_relative_to_goal > 0.5:  # Significant reward when near opponent's goal
                components["ball_control_reward"][rew_index] += 0.2 * ball_position_relative_to_goal

            reward[rew_index] = reward[rew_index] + components["ball_control_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Update this to handle both team actions and ball possession status
        if obs is not None:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] += action
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
