import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward based on the dynamics of Sprint and Stop-Sprint actions for optimizing defensive coverage."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking how many times each sticky action is used

    def reset(self):
        """Reset the sticky actions counter on reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the current state of the reward wrapper."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the reward wrapper."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Adjust the reward based on the use of Sprint and Stop-Sprint actions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        # Assuming 'sticky_actions' indices 8 & 9 correspond to 'sprint' & 'stop-sprint' actions
        sprint_action_index = 8
        stop_sprint_action_index = 9

        # Coefficients for sprint usage and stop-sprint usage
        sprint_coeff = 0.1
        stop_sprint_coeff = 0.05

        # Reward modification parts
        sprint_reward = [0.0]
        stop_sprint_reward = [0.0]

        if observation['sticky_actions'][sprint_action_index] == 1:
            self.sticky_actions_counter[sprint_action_index] += 1
            sprint_reward[0] += sprint_coeff
        
        if observation['sticky_actions'][stop_sprint_action_index] == 1:
            self.sticky_actions_counter[stop_sprint_action_index] += 1
            stop_sprint_reward[0] += stop_sprint_coeff

        reward[0] += sprint_reward[0] + stop_sprint_reward[0]
        components['sprint_reward'] = sprint_reward
        components['stop_sprint_reward'] = stop_sprint_reward

        return reward, components

    def step(self, action):
        """Overwrites the step function to include custom reward adjustments."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update info with sticky_actions usage
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_taken in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_taken

        return observation, reward, done, info
