import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense, strategy-oriented reward for developing offensive skills."""

    def __init__(self, env):
        super().__init__(env)
        self.passing_accuracy_bonus = 0.1
        self.dribbling_bonus = 0.2
        self.shooting_accuracy_bonus = 0.3
        self.offensive_position_bonus = 0.05

    def reset(self):
        """Reset the environment and clean up any state."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment to save the current simulation state."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment to load a previous simulation state."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Custom reward function focused on offensive football strategies."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_accuracy": [0.0] * len(reward),
                      "dribbling": [0.0] * len(reward),
                      "shooting_accuracy": [0.0] * len(reward),
                      "offensive_position": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx, obs in enumerate(observation):
            # Reward for successful passing strategy using ball possession status
            if obs['ball_owned_team'] == 0:  # Assuming 0 is the controlled team
                components["passing_accuracy"][idx] = self.passing_accuracy_bonus
                reward[idx] += components["passing_accuracy"][idx]

            # Reward for effective dribbling checked via sticky actions
            if obs.get('sticky_actions', [])[9] == 1:  # Assuming index 9 is dribbling action
                components["dribbling"][idx] = self.dribbling_bonus
                reward[idx] += components["dribbling"][idx]

            # Reward for shooting towards the goal
            goal_distance = np.abs(obs['ball'][0] - 1)  # Only X position towards opponent's goal
            if goal_distance < 0.1:  # Close to the goal on X-axis
                components["shooting_accuracy"][idx] = self.shooting_accuracy_bonus * (0.1 - goal_distance)
                reward[idx] += components["shooting_accuracy"][idx]

            # Reward based on offensive position
            if obs['ball_owned_team'] == 0 and obs['ball'][0] >= 0.5:  # Player is in the offensive half
                components["offensive_position"][idx] = self.offensive_position_bonus
                reward[idx] += components["offensive_position"][idx]

        return reward, components

    def step(self, action):
        """Step the environment by applying an action, modifying the reward, and returning results."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Provide combined information about the final reward and individual components
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
