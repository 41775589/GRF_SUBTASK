import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized midfield dynamics reward."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.role_dynamics_reward = 0.05
        self.central_midfielder_indices = [4, 5]  # Typically central midfield roles
        self.wide_midfielder_indices = [6, 7]     # Typically wide midfield roles
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "midfield_dynamics_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
          
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            roles = o.get('left_team_roles', [])
            central_mid_presence = any(role in self.central_midfielder_indices for role in roles)
            wide_mid_presence = any(role in self.wide_midfielder_indices for role in roles)

            if central_mid_presence and wide_mid_presence:
                components["midfield_dynamics_reward"][rew_index] = self.role_dynamics_reward
                reward[rew_index] += components["midfield_dynamics_reward"][rew_index]

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
