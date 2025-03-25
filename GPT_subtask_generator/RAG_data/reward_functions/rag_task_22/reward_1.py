import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on sprinting to improve defensive coverage."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int) # Track usage of each sticky action
        self.defensive_positions = np.linspace(-1, 1, 10)  # Positions along the x-axis of the pitch
        self.position_rewards = np.zeros(10)  # Rewards for reaching positions along the field
        self.reward_scale = 0.05  # Scale for rewards at different defensive checkpoints

    def reset(self):
        """Reset the environment and reward tracking."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the current environment state with wrapper specific augmentations."""
        to_pickle['CheckpointRewardWrapper'] = self.position_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the environment state with specific augmentations for the wrapper."""
        from_pickle = self.env.set_state(state)
        self.position_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Compute the reward using position based dense checkpoints for improving defensive coverage."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "defensive_position_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components
        
        for index, o in enumerate(observation):
            # Check the position and sprint action (index 8 in sticky_actions)
            x_position = o['left_team'][o['active']][0]
            if o['sticky_actions'][8]:
                target_indices = np.where((self.defensive_positions - x_position) ** 2 < 0.01)[0]
                if len(target_indices) > 0:
                    reward_index = target_indices[0]
                    if self.position_rewards[reward_index] == 0:
                        components["defensive_position_reward"][index] = self.reward_scale
                        reward[index] += components["defensive_position_reward"][index]
                        self.position_rewards[reward_index] = 1  # Mark this position as rewarded

        return reward, components

    def step(self, action):
        """Step the environment, return observations, reward, done, and info augmented with rewards components."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
