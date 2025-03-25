import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for specific player behaviors in football,
    focusing on skills crucial for midfielders and defenders like high passes,
    long passes, dribbling, and sprint management."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_reward = 0.2
        self.long_pass_reward = 0.2
        self.dribble_reward = 0.1
        self.sprint_reward = 0.05
        self.sprint_stop_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "long_pass_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward),
                      "sprint_stop_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward for executing high and long passes
            if o['sticky_actions'][8]:  # High pass action index
                components["high_pass_reward"][rew_index] = self.high_pass_reward
            if o['sticky_actions'][7]:  # Long pass action index
                components["long_pass_reward"][rew_index] = self.long_pass_reward

            # Reward for dribbling effectively
            if o['sticky_actions'][9]:  # Dribble action index
                components["dribble_reward"][rew_index] = self.dribble_reward

            # Sprint management
            if o['sticky_actions'][4]:  # Sprint action index
                components["sprint_reward"][rew_index] = self.sprint_reward
            if self.sticky_actions_counter[4] and not o['sticky_actions'][4]:  # Stop sprint
                components["sprint_stop_reward"][rew_index] = self.sprint_stop_reward

            # Calculate total reward for the current index
            total_component_reward = sum([components[key][rew_index] for key in components if key != "base_score_reward"])
            reward[rew_index] += total_component_reward

        self.sticky_actions_counter = np.array([o['sticky_actions'] for o in observation]).sum(axis=0)
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
