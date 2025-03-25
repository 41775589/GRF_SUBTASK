import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the rewards emitted from FootballEnv to promote fast-paced and precision-based offense strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters
        self.precision_reward_factor = 0.2
        self.fast_break_reward = 0.5
        self.scoring_window = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return super().reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        components = {
            "base_score_reward": reward.copy(),
            "precision_reward": [0.0] * len(reward),
            "fast_break_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for scoring within a certain range to the goal
            if 'score' in o and reward[rew_index] == 1:
                if abs(o['ball'][0]) > 1 - self.scoring_window:
                    components["precision_reward"][rew_index] = self.precision_reward_factor
                reward[rew_index] += components["precision_reward"][rew_index]
            
            # Reward agents for fast play: if they manage to reach the opponent’s half quickly
            if 'ball' in o and o['ball'][0] > 0.5:
                components["fast_break_reward"][rew_index] = self.fast_break_reward
                reward[rew_index] += components["fast_break_reward"][rew_index]
        
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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
