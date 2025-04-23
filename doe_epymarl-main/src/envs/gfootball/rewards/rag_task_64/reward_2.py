import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful high passes and crossings to promote spatial creation in attack."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality_reward = 0.1
        self.cross_effectiveness_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_quality_reward": [0.0] * len(reward),
            "cross_effectiveness_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Encourage high passes
            if o['ball_owned_team'] == 0 and 'ball' in o and o['ball'][2] > 0.3:
                components["pass_quality_reward"][rew_index] = self.pass_quality_reward
                reward[rew_index] += self.pass_quality_reward

            # Encourage effective crosses into the box
            if o['ball_owned_team'] == 0 and -0.3 <= o['ball'][1] <= 0.3 and abs(o['ball'][0]) > 0.7:
                components["cross_effectiveness_reward"][rew_index] = self.cross_effectiveness_reward
                reward[rew_index] += self.cross_effectiveness_reward

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
