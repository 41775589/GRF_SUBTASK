import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for efficient use of Stop-Sprint and Stop-Moving actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset sticky actions counter and environment on reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Helper function to get current state of the environment for serialization."""
        to_pickle['CheckpointRewardWrapper'] = {}  # Put state-specific items here if needed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Helper function to set the state of the environment from deserialized data."""
        from_pickle = self.env.set_state(state)
        # Apply state if any specific to this wrapper.
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on usage of Sprint and Stop actions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        for agent_index, obs in enumerate(observation):
            sprint_action = obs["sticky_actions"][8]  # Index of sprint action in sticky actions
            dribble_action = obs["sticky_actions"][9]  # Index of dribble action in sticky actions

            components.setdefault("sprint_reward", [0.0, 0.0])
            components.setdefault("stop_reward", [0.0, 0.0])
            
            # Encourage stopping sprint for stamina conservation
            if sprint_action == 0 and self.sticky_actions_counter[8] > 0:
                components["stop_reward"][agent_index] += 0.05  
                reward[agent_index] += components["stop_reward"][agent_index]
            
            # Reduce reward if sprint is continuously used, as it may tire out the player
            if sprint_action == 1:
                components["sprint_reward"][agent_index] -= 0.02  
                reward[agent_index] += components["sprint_reward"][agent_index]

            # Update sticky actions counter for sprinting
            self.sticky_actions_counter[8] += sprint_action
            self.sticky_actions_counter[9] += dribble_action

        return reward, components

    def step(self, action):
        """Take a step using the underlying environment and apply wrapped reward modifications."""
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
