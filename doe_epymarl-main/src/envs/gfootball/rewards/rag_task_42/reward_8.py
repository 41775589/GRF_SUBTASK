import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances training by focusing on midfield gameplay dynamics, coordination under pressure,
    and strategic positioning for effective transition between offense and defense."""

    def __init__(self, env):
        super().__init__(env)
        self.checkpoint_count = 10  # Number of checkpoint zones across the midfield
        self.checkpoints_collected = {}
        self.checkpoint_reward = 0.1  # Reward increment for hitting new checkpoints
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Sticky actions register
        
    def reset(self):
        """Resetting for new game episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoints_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the internal state for serialization."""
        to_pickle['CheckpointRewardWrapper'] = self.checkpoints_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the internal state from deserialized data."""
        from_pickle = self.env.set_state(state)
        self.checkpoints_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Modify reward based on midfield control and strategic transitioning."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_dynamic_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Calculate distance from the center of the field (0 in x-axis)
            midfield_distance = abs(o['right_team'][o['active']][0])  # x coordinate of active player

            # Calculate checkpoint index based on player's x position
            checkpoint_index = min(int(midfield_distance * self.checkpoint_count), self.checkpoint_count - 1)

            # Collect rewards for reaching new checkpoints that haven't been visited
            if checkpoint_index not previously visited:
                components["midfield_dynamic_reward"][rew_index] = self.checkpoint_reward
                reward[rew_index] += self.checkpoint_reward

                # Mark this checkpoint as collected
                self.checkpoints_collected[rew_index] = checkpoint_index
        
        return reward, components

    def step(self, action):
        """Apply an action, and augment output with custom reward details."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Aggregate the reward components for logging
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Ensure sticky action recording
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
