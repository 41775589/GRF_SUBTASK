import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for aggressive defensive tactics such as tackles and slides."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_reward = 0.5
        self.slide_reward = 1.0
        self.opponent_control_penalty = -0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['DefensiveTactics'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['DefensiveTactics'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "slide_reward": [0.0] * len(reward),
                      "opponent_control_penalty": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            reward[rew_index] += components["base_score_reward"][rew_index]

            sticky_actions = o['sticky_actions']
            # Checking for tackles (action index e.g., 1) and slides (action index e.g., 8).
            if sticky_actions[1]:  # Assuming index 1 is tackle.
                reward[rew_index] += self.tackle_reward
                components["tackle_reward"][rew_index] += self.tackle_reward

            if sticky_actions[8]:  # Assuming index 8 is slide.
                reward[rew_index] += self.slide_reward
                components["slide_reward"][rew_index] += self.slide_reward

            # Penalty for ball being controlled by opponent near our goal area (simplistic scenario).
            if o['ball_owned_team'] == 1 and abs(o['ball'][0] + 1) < 0.3:  # Simulating closeness to left team's goal
                reward[rew_index] += self.opponent_control_penalty
                components["opponent_control_penalty"][rew_index] += self.opponent_control_penalty

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
        return observation, reward, done, info
