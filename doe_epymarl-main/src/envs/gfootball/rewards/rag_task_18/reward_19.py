import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a midfield synergy and pace control reward."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize variables for tracking midfield control and pace
        self.midfield_pace_bonus = 0.05
        self.passing_efficiency_bonus = 0.03
        self.control_zones = np.linspace(-0.5, 0.5, 10)  # Define zones in midfield
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(), "midfield_pace_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            ball_position = o['ball'][0]  # Use X-coordinate for lateral movement
            team_controls_midfield = o['left_team_direction'][:, 0].mean()
            team_movement = o['left_team_direction'][:, 0]  # Use X-direction for pace

            control_qualification = any(zone - 0.1 < ball_position < zone + 0.1 for zone in self.control_zones)
            if control_qualification and team_controls_midfield > 0:
                components["midfield_pace_reward"][rew_index] += self.midfield_pace_bonus
                reward[rew_index] += components["midfield_pace_reward"][rew_index]

            if np.abs(team_movement).mean() < 0.01:  # Encouraging controlled pace
                components["midfield_pace_reward"][rew_index] += self.passing_efficiency_bonus
                reward[rew_index] += components["midfield_pace_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
