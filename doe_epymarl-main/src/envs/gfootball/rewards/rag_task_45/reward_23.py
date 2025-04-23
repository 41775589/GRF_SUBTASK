import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering stop-sprint and stop-moving techniques."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {
            "base_score_reward": reward.copy(),
            "stop_sprint_reward": [0.0] * len(reward),
            "stop_move_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            actions = obs.get('sticky_actions', [])
            previous_actions = self.sticky_actions_counter

            # Reward for stopping from sprint
            if previous_actions[8] == 1 and actions[8] == 0:
                components["stop_sprint_reward"][rew_index] = 1.0
                reward[rew_index] += components["stop_sprint_reward"][rew_index]

            # Reward for moving to stopping
            if np.any(previous_actions[:8]) and np.all(actions[:8] == 0):
                components["stop_move_reward"][rew_index] = 1.0
                reward[rew_index] += components["stop_move_reward"][rew_index]

        self.sticky_actions_counter = actions
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
