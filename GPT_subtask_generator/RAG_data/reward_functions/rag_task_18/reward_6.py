import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances coordination and control in midfield play."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Define parameters for midfield coordination and pace control
        self.midfield_threshold = 0.2  # Threshold to consider being in the midfield
        self.pace_threshold = 0.1  # Threshold for pace changes to be considered significant
        self.midfield_control_reward = 0.2
        self.pace_management_reward = 0.1

        # Stores the previous pace to calculate pace change
        self.prev_pace = {}

    def reset(self):
        """Reset the environment and the wrapper components."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_pace = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the wrapper state for saving."""
        state = self.env.get_state(to_pickle)
        state['prev_pace'] = self.prev_pace
        return state

    def set_state(self, state):
        """Set the wrapper state for loading."""
        self.prev_pace = state['prev_pace']
        return self.env.set_state(state)

    def reward(self, reward):
        """Custom reward function focusing on midfield control and pace management."""
        observation = self.env.unwrapped.observation()
        
        components = {
            "base_score_reward": reward.copy(),
            "midfield_control_reward": [0.0] * len(reward),
            "pace_management_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            # Midfield control reward
            if abs(o['ball'][0]) < self.midfield_threshold:
                components["midfield_control_reward"][i] = self.midfield_control_reward
                reward[i] += components["midfield_control_reward"][i]

            # Pace management reward
            pace = np.linalg.norm(o['ball_direction'])
            if i in self.prev_pace and abs(pace - self.prev_pace[i]) > self.pace_threshold:
                components["pace_management_reward"][i] = self.pace_management_reward
                reward[i] += components["pace_management_reward"][i]
            
            self.prev_pace[i] = pace

        return reward, components

    def step(self, action):
        """Take a step in the environment applying the new reward scheme."""

        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Add rewards breakdown to info for better transparency
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Track and update sticky actions for each agent
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
