import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for long-range shooting skills."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._outside_penalty_box_reward = 0.3
        self._shot_taken_reward = 0.5

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
                      "outside_penalty_box_reward": [0.0] * len(reward),
                      "shot_taken_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, obs in enumerate(observation):
            # Reward for being outside the penalty box [distance from center greater than certain threshold]
            pos_x = obs['ball'][0]
            if abs(pos_x) > 0.7:
                components["outside_penalty_box_reward"][index] = self._outside_penalty_box_reward
                reward[index] += components["outside_penalty_box_reward"][index]

            if any(obs['sticky_actions'][4:7]):  # Assumes indices 4, 5, 6 correspond to shooting actions
                components["shot_taken_reward"][index] = self._shot_taken_reward
                reward[index] += components["shot_taken_reward"][index]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
