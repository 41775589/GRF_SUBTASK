import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on tackles."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.tackle_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_actions = [None] * 2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_actions = [None] * 2
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.prev_actions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.prev_actions = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_actions = o['sticky_actions']
            prev_actions = self.prev_actions[rew_index]

            # Check for sliding (action 10) and standing (action 11) tackles.
            # Assume 'action_sliding_tackle' and 'action_standing_tackle' map to indices 10 and 11
            if prev_actions is not None:
                if current_actions[10] and not prev_actions[10]:
                    components["tackle_reward"][rew_index] = self.tackle_reward
                if current_actions[11] and not prev_actions[11]:
                    components["tackle_reward"][rew_index] += self.tackle_reward

            reward[rew_index] += components["tackle_reward"][rew_index]
            self.prev_actions[rew_index] = current_actions.copy()

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
