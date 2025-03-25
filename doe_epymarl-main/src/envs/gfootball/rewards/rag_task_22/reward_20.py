import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for sprint actions to improve defensive coverage.
    This reward is intended to encourage faster repositioning across the field.
    """
    def __init__(self, env):
        super().__init__(env)
        # Initialize sprint action count tracker
        self.sprint_actions_counter = np.zeros(10, dtype=int)
        # Reward increase factors
        self.sprint_reward_increase = 0.02

    def reset(self):
        # Reset the counter on each reset
        self.sprint_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sprint_actions_counter'] = self.sprint_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sprint_actions_counter = from_pickle.get('sprint_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        # Access the current observations from the environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Give extra rewards for using sprint actions
            if o['sticky_actions'][8] == 1:  # Index 8 corresponds to sprint action in sticky_actions
                components["sprint_reward"][rew_index] += self.sprint_reward_increase
                reward[rew_index] += components["sprint_reward"][rew_index]
                self.sprint_actions_counter[rew_index] += 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
