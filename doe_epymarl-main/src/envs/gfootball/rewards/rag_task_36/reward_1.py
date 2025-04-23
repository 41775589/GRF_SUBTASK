import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on dribbling and dynamic positioning."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_start_reward = 0.1
        self.dribble_stop_reward = 0.05
        self.position_change_reward = 0.02

    def reset(self):
        """Reset sticky actions counter and environment states."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Include additional wrapper state when saving the environment state."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore additional wrapper state when loading the environment state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        """Modify the reward function based on dribbling and dynamic positioning."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribble_start_reward": [0.0] * len(reward),
            "dribble_stop_reward": [0.0] * len(reward),
            "position_change_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for starting dribble
            if o['sticky_actions'][9] == 1 and self.sticky_actions_counter[9] == 0:
                components["dribble_start_reward"][rew_index] = self.dribble_start_reward
                reward[rew_index] += components["dribble_start_reward"][rew_index]

            # Reward for stopping dribble
            if o['sticky_actions'][9] == 0 and self.sticky_actions_counter[9] == 1:
                components["dribble_stop_reward"][rew_index] = self.dribble_stop_reward
                reward[rew_index] += components["dribble_stop_reward"][rew_index]

            # Track and reward position change
            if (self.sticky_actions_counter[:4] != o['sticky_actions'][:4]).any():
                components["position_change_reward"][rew_index] = self.position_change_reward
                reward[rew_index] += components["position_change_reward"][rew_index]

            self.sticky_actions_counter = o['sticky_actions'].copy()

        return reward, components

    def step(self, action):
        """Wrap the environment's step function to modify the reward."""
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
