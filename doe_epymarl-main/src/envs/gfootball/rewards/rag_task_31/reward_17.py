import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defensive actions such as tackling and sliding, optimized for quick responses."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_actions_rewards = 0.1  # Incremental reward for tackling or sliding
        self.defensive_action_initialized = False

    def reset(self):
        """Reset the environment and counters for sticky actions."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_action_initialized = False
        return self.env.reset()
 
    def get_state(self, to_pickle):
        """Get observation state with current state of rewards."""
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist(),
            'defensive_action_initialized': self.defensive_action_initialized
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set state of the environment from pickle and initialize fields related to rewards."""
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = np.array(state_data['sticky_actions_counter'])
        self.defensive_action_initialized = state_data['defensive_action_initialized']
        return from_pickle

    def reward(self, reward):
        """Custom reward function that incentivizes prompt defensive actions."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_bonus_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for idx, obs in enumerate(observation):
            if self.defensive_action_initialized:
                continue

            # Check for tackling or sliding, corresponding to action indices in sticky_actions
            is_defensive_action = obs['sticky_actions'][8] or obs['sticky_actions'][9]  # Typically indices for slide tackle

            if is_defensive_action:
                components["defensive_bonus_reward"][idx] = self.defensive_actions_rewards
                reward[idx] += components["defensive_bonus_reward"][idx]
                self.defensive_action_initialized = True
            
        return reward, components

    def step(self, action):
        """Step function ensuring the dense rewards are properly tracked along with standard steps."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        for i, action_active in enumerate(self.sticky_actions_counter):
            info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
