import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for defensive actions like tackling and sliding."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_reward = 0.2
        self.slide_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, rewards):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(rewards).copy(),
                      "tackle_reward": np.zeros(len(rewards)),
                      "slide_reward": np.zeros(len(rewards))}

        for i, (rew, o) in enumerate(zip(rewards, observation)):
            sticky_actions = o['sticky_actions']
            if sticky_actions[3] == 1:  # tackle action is active
                components["tackle_reward"][i] = self.tackle_reward
            if sticky_actions[4] == 1:  # slide action is active
                components["slide_reward"][i] = self.slide_reward
            rewards[i] += components["tackle_reward"][i] + components["slide_reward"][i]
        
        return rewards, components

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
