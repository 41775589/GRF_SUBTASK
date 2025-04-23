import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on defensive maneuvers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_reward = 0.5
        self.prev_sticky_actions = [0] * len(self.env.action_space)
        self.transition_count = [0] * len(self.env.action_space)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_sticky_actions = [0] * len(self.env.action_space)
        self.transition_count = [0] * len(self.env.action_space)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['passes_transition_counter'] = self.transition_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.transition_count = from_pickle['passes_transition_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        modified_reward = reward.copy()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            current_sticky_actions = obs['sticky_actions']
            
            crossing_actions_indices = [0, 3, 7]  # indexes for left, right, bottom_left actions
            transitions = [abs(prev - curr) for prev, curr in zip(self.prev_sticky_actions, current_sticky_actions)]
            
            self.prev_sticky_actions = current_sticky_actions
            
            # A reward is given if there is a change from standing to moving in a defensive mode and vice versa
            if any(current_sticky_actions[i] != self.prev_sticky_actions[i] for i in crossing_actions_indices):
                num_changes = sum(transitions)
                # Apply a diminishing reward based on frequency of transitions to discourage excessive stopping/starting
                factor = max(0, 1 - self.transition_count[i] * 0.1)
                transition_bonus = factor * num_changes * self.transition_reward
                modified_reward[i] += transition_bonus
                components['transition_reward'][i] = transition_bonus
                self.transition_count[i] += 1

        return modified_reward, components

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
