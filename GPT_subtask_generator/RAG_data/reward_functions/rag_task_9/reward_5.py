import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that refines reward calculation by focusing on offensive skills such as passing, shooting, and dribbling."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_counter = 0
        self.shooting_counter = 0
        self.dribbling_counter = 0
        self.passing_reward = 0.05
        self.shooting_reward = 0.1
        self.dribbling_reward = 0.03

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_counter = 0
        self.shooting_counter = 0
        self.dribbling_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'passing_counter': self.passing_counter,
            'shooting_counter': self.shooting_counter,
            'dribbling_counter': self.dribbling_counter,
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.passing_counter = from_pickle['CheckpointRewardWrapper']['passing_counter']
        self.shooting_counter = from_pickle['CheckpointRewardWrapper']['shooting_counter']
        self.dribbling_counter = from_pickle['CheckpointRewardWrapper']['dribbling_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            # Reward for passing action
            if o['sticky_actions'][7]:  # Assuming index 7 corresponds to "pass"
                components["passing_reward"][rew_index] = self.passing_reward
                reward[rew_index] += components["passing_reward"][rew_index]

            # Reward for shooting action
            if o['sticky_actions'][4]:  # Assuming index 4 corresponds to "shoot"
                components["shooting_reward"][rew_index] = self.shooting_reward
                reward[rew_index] += components["shooting_reward"][rew_index]

            # Reward for dribbling
            if o['sticky_actions'][9]:  # Assuming index 9 corresponds to "dribble"
                components["dribbling_reward"][rew_index] = self.dribbling_reward
                reward[rew_index] += components["dribbling_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Reset sticky actions counter
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_state
                info[f"sticky_actions_{i}"] = action_state
        return observation, reward, done, info
