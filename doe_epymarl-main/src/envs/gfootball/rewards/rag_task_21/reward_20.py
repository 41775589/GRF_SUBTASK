import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense defensive reward for intercepting and positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_zones = 10
        self._intercept_reward = 0.05
        self._positioning_reward = 0.05

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
                      "intercept_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['ball_owned_team'] == 0 and o['designated'] == o['active']:  # Player's team intercepts the ball
                components["intercept_reward"][rew_index] = self._intercept_reward
                reward[rew_index] += 1.2 * components["intercept_reward"][rew_index]
            
            # Defensive positioning reward: encouraging being closer to own goal when not possessing the ball
            if o['ball_owned_team'] != 0 and o['designated'] == o['active']:
                distance_to_goal = np.linalg.norm(o['left_team'][o['active']] + np.array([1.0, 0.0]))
                components["positioning_reward"][rew_index] = self._positioning_reward * (1 - np.tanh(distance_to_goal))
                reward[rew_index] += 1.1 * components["positioning_reward"][rew_index]

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
            for action_index, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[action_index] = action
        return observation, reward, done, info
