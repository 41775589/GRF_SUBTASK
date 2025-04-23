import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a complex reward for offensive football strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_reward = 1.0
        self.dribbling_reward = 0.5
        self.passing_reward = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "shooting": [0.0] * len(reward),
                      "dribbling": [0.0] * len(reward),
                      "passing": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Evaluate shooting by checking ball direction speeds and shooting attempts
            if o['sticky_actions'][6] and o['ball_owned_team'] == 0:  # Assuming action 6 represents shooting.
                components["shooting"][rew_index] = self.shooting_reward
                reward[rew_index] += components["shooting"][rew_index]

            # Evaluate dribbling by monitoring dribble activation 
            if o['sticky_actions'][9]:  # Assuming action 9 represents dribbling.
                components["dribbling"][rew_index] = self.dribbling_reward
                reward[rew_index] += components["dribbling"][rew_index]

            # Evaluate passing, especially long or high passes
            if (o['sticky_actions'][2] or o['sticky_actions'][3]) and o['ball_owned_team'] == 0:  # Assuming actions 2,3 are long/high passes.
                components["passing"][rew_index] = self.passing_reward
                reward[rew_index] += components["passing"][rew_index]

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
