import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on energy conservation tactics."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Maintain the stop and sprint counters for each agent
        self.stop_sprint_counters = np.zeros((2,), dtype=int)
        self.stop_moving_counters = np.zeros((2,), dtype=int)
        self.reward_for_stop_sprint = 0.05
        self.reward_for_stop_movement = 0.03

    def reset(self):
        self.stop_sprint_counters.fill(0)
        self.stop_moving_counters.fill(0)
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "stop_sprint_reward": np.zeros(2, dtype=float),
                      "stop_movement_reward": np.zeros(2, dtype=float)}
        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            # Check stop_sprint usage
            if o['sticky_actions'][8] == 0:  # 'action_sprint' is at index 8
                self.stop_sprint_counters[idx] += 1
            else:
                self.stop_sprint_counters[idx] = 0
            
            # Check stop_movement
            if np.all(o['sticky_actions'][0:8] == 0):  # movement actions are the first 8
                self.stop_moving_counters[idx] += 1
            else:
                self.stop_moving_counters[idx] = 0

            # Calculate rewards
            components['stop_sprint_reward'][idx] = self.stop_sprint_counters[idx] * self.reward_for_stop_sprint
            components['stop_movement_reward'][idx] = self.stop_moving_counters[idx] * self.reward_for_stop_movement
            
            # Add the new rewards to the original game rewards
            reward[idx] += components['stop_sprint_reward'][idx] + components['stop_movement_reward'][idx]

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
