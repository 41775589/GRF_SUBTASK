import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides a reward for enhancing shot precision skills in tight spaces.
    This focuses on rewarding controlled actions that improve angles and power required to beat the goalkeeper.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.precision_thresholds = np.linspace(0.1, 0.9, 5)  # Define precision thresholds
        self.precision_rewards = np.linspace(0.1, 1.0, 5)     # Rewards for meeting each threshold

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment, including wrapped components."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment based on the state provided."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Augments the incoming reward with additional rewards for precision in tight spaces.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "precision_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            distance_to_goal = np.sqrt((o['ball'][0] - 1)**2 + (o['ball'][1]**2))
            angle_to_goal = np.arctan2(abs(o['ball'][1]), 1 - abs(o['ball'][0]))  # Angle with respect to exact goal position

            # Check for each threshold if the current action respects the desired precision requirements
            precision_meets_threshold = [dist <= threshold for dist, threshold in zip([distance_to_goal, angle_to_goal], self.precision_thresholds)]
            sum_rewards_for_precision = np.dot(precision_meets_threshold, self.precision_rewards)
            reward[rew_index] += sum_rewards_for_precision  # Add up rewards corresponding to thresholds met

            components["precision_reward"][rew_index] = sum_rewards_for_precision

        return reward, components

    def step(self, action):
        """Apply the action and adjust rewards incorporating additional details into info dict."""
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
