import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for mastering high passes and expanding gameplay on the width of the field."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._wide_play_bonus = 0.1
        self._high_pass_bonus = 0.2
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
                      "wide_play_bonus": [0.0] * len(reward),
                      "high_pass_bonus": [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Wide play bonus when player is near side boundaries
            if abs(o['left_team'][o['active']][1]) > 0.3:  # Y positions near boundaries
                components["wide_play_bonus"][rew_index] = self._wide_play_bonus
                reward[rew_index] += components["wide_play_bonus"][rew_index]
            
            # High pass bonus: Special action index (assuming here that index 5 might be a high pass)
            if 'high_pass' in o['sticky_actions'] and o['sticky_actions'][5] == 1:
                components["high_pass_bonus"][rew_index] = self._high_pass_bonus
                reward[rew_index] += components["high_pass_bonus"][rew_index]
                
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
            index = 0
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[index] = action
                index += 1
        return observation, reward, done, info
