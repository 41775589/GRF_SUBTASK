import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a sophisticated reward based on possession changes and strategic positioning."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._collected_rewards = {}
        self._num_zones = 5
        self._possession_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["possession_reward"][rew_index] = 0.0
            
            if o['ball_owned_team'] == -1:
                # Ball not owned by any team, reset possession flag
                self._collected_rewards[rew_index] = 'none'
            elif o['ball_owned_team'] == 0:
                if self._collected_rewards.get(rew_index, 'none') == 'right':
                    # Ball ownership changed from right team to left team
                    components["possession_reward"][rew_index] = self._possession_reward
                    reward[rew_index] += components["possession_reward"][rew_index]
                self._collected_rewards[rew_index] = 'left'
            elif o['ball_owned_team'] == 1:
                if self._collected_rewards.get(rew_index, 'none') == 'left':
                    # Ball ownership changed from left team to right team
                    components["possession_reward"][rew_index] = self._possession_reward
                    reward[rew_index] += components["possession_reward"][rew_index]
                self._collected_rewards[rew_index] = 'right'
        
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
