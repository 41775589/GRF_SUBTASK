import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for defensive actions like tackling and aggressive maneuvers such as sliding. 
    Optimizes for fast response times to opponent attacks."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._tackles_attempts = {0: 0, 1: 0}
        self._successful_tackles = {0: 0, 1: 0}
        self._aggressive_actions = {0: 0, 1: 0}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self._tackles_attempts = {0: 0, 1: 0}
        self._successful_tackles = {0: 0, 1: 0}
        self._aggressive_actions = {0: 0, 1: 0}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['DefensiveStatistics'] = {
            'tackles_attempts': self._tackles_attempts,
            'successful_tackles': self._successful_tackles,
            'aggressive_actions': self._aggressive_actions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        stats = from_pickle['DefensiveStatistics']
        self._tackles_attempts = stats['tackles_attempts']
        self._successful_tackles = stats['successful_tackles']
        self._aggressive_actions = stats['aggressive_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            # Reward for tackles
            if 'action_sprint' in obs['sticky_actions']:
                self._tackles_attempts[rew_index] += 1
                components['defensive_reward'][rew_index] += 0.02

            # Reward for successful ball recovery
            if obs['ball_owned_team'] == rew_index:
                self._successful_tackles[rew_index] += 1
                components['defensive_reward'][rew_index] += 0.05

            # Reward for aggressive actions like sliding
            if 'action_sliding' in obs['sticky_actions']:
                self._aggressive_actions[rew_index] += 1
                components["defensive_reward"][rew_index] += 0.03

        reward = [reward[idx] + components["defensive_reward"][idx] for idx in range(len(reward))]
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Reset the sticky actions count every step
        self.sticky_actions_counter.fill(0)
        # Update for each action:
        for agent_obs in obs:
            for i, act in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = act
        return observation, reward, done, info
