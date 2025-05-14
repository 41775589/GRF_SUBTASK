import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on defensive actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sliding_coefficient = 2.0
        self.sprinting_coefficient = 1.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Assuming the specific set of sticky actions

    def reset(self):
        """Reset the sticky actions counter when the environment is reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Pickle the current state along with CheckpointRewardWrapper information."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the internal state from a pickle."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on defensive actions performed by agents."""
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "sliding_reward": [0.0],
            "sprinting_reward": [0.0]
        }

        for o in observation:
            # Check for sliding action
            if 'sticky_actions' in o and o['sticky_actions'][8] == 1:  # Index 8 for sliding
                components["sliding_reward"][0] = self.sliding_coefficient
            # Check for sprinting action
            if 'sticky_actions' in o and o['sticky_actions'][6] == 1:  # Index 6 for sprinting
                components["sprinting_reward"][0] = self.sprinting_coefficient
            
        # Calculate total reward by considering the base score reward and additional rewards
        reward[0] = (1 * components["base_score_reward"][0] + components["sliding_reward"][0] 
                     + components["sprinting_reward"][0])

        return reward, components

    def step(self, action):
        """Capture the action and its outcome, modify the reward accordingly, and return modified observations and info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update info with sticky actions count
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
