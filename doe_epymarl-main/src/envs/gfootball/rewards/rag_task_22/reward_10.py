import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focused on enhancing sprint-related defensive movements."""

    def __init__(self, env):
        super().__init__(env)
        self.sprint_usage_counts = [0, 0]  # Counts sprint usages for each agent
        self.max_sprint_usage = 10
        self.sprint_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_usage_counts = [0, 0]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sprint_usage_counts'] = self.sprint_usage_counts
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        state_dict = self.env.set_state(state)
        self.sprint_usage_counts = state_dict.get('sprint_usage_counts', [0, 0])
        return state_dict

    def reward(self, reward):
        # Capture the current state from the wrapped environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward)}

        for i, o in enumerate(observation):
            # Check if the sprint action was taken
            if o['sticky_actions'][8] == 1:
                # Reward agents that use the sprint effectively
                if self.sprint_usage_counts[i] < self.max_sprint_usage:
                    components["sprint_reward"][i] = self.sprint_reward
                    reward[i] += components["sprint_reward"][i]
                self.sprint_usage_counts[i] += 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for idx, act in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{idx}"] = act
        return observation, reward, done, info
