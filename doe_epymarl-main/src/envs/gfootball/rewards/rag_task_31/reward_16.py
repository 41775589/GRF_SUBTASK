import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that enhances the reward function with a focus on defensive actions such as tackling and sliding. """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_action_counters'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_action_counters']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        assert len(reward) == len(observation)

        # Custom reward adjustment based on defensive actions
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Get the current action sticky states
            sticky_actions = o['sticky_actions']
            tackling_action = 0.3  # A higher reward for successful tackling
            sliding_action = 0.5  # Reward sliding tackles even more

            if sticky_actions[0]:  # Assuming index 0 is tackling
                reward[rew_index] += tackling_action
                self.sticky_actions_counter[0] += 1

            if sticky_actions[1]:  # Assuming index 1 is sliding
                reward[rew_index] += sliding_action
                self.sticky_actions_counter[1] += 1

            components['defensive_action_reward'] = [tackling_action if sticky_actions[0] else 0.0 for _ in reward]

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
            for i, active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = active
        return observation, reward, done, info
