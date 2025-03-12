import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for mastering defensive maneuvers particularly focusing on sliding tackles under high-pressure situations."""

    def __init__(self, env):
        super().__init__(env)
        self.sliding_tackle_reward = 0.5  # Reward contribution from successful sliding tackles
        self.high_pressure_defense_reward = 0.2  # Additional reward for defending under high pressure

    def reset(self):
        """Resets the environment and clears any internal data."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Encapsulate the state of the environment for future restoration."""
        to_pickle['CheckpointRewardWrapper'] = {}  # Add internal states if any
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the environment from the saved state."""
        from_pickle = self.env.set_state(state)
        # Restore any internal states here if necessary from from_pickle
        return from_pickle

    def reward(self, reward):
        """Specialized reward function to encourage sliding tackles under high-pressure scenarios."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "sliding_tackle_reward": [0.0] * len(reward),
            "pressure_defense_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if the agent performed a sliding tackle successfully
            if 'sticky_actions' in o and o['sticky_actions'][7] == 1:  # Assuming index 7 is sliding
                components["sliding_tackle_reward"][rew_index] = self.sliding_tackle_reward
            
            # Check for defensive actions under high-pressure (close proximity to goal or multiple attackers)
            if 'right_team' in o and np.min(o['right_team'][:, 0]) < -0.5:  # Positions closer to goal
                components["pressure_defense_reward"][rew_index] = self.high_pressure_defense_reward

            # Calculate the total reward
            reward[rew_index] = components["base_score_reward"][rew_index]
            reward[rew_index] += components["sliding_tackle_reward"][rew_index]
            reward[rew_index] += components["pressure_defense_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Calls the original step method and updates the reward using specialized reward function."""
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
