import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that implements a task-specific reward for training agents to excel in quick decision-making
    and efficient ball handling to initiate counter-attacks immediately after recovering possession.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owned_team = -1  # To track changes in ball possession

    def reset(self):
        """
        Reset the environment and reward tracking metrics.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owned_team = -1
        return self.env.reset()

    def reward(self, reward):
        """
        Modifies the reward based on ball possession changes to encourage efficient counter-attacks.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_change_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            current_obs = observation[idx]
            if current_obs['ball_owned_team'] != self.previous_ball_owned_team:
                if self.previous_ball_owned_team == -1 and current_obs['ball_owned_team'] == 0:
                    # Reward for gaining possession
                    components["possession_change_reward"][idx] = 0.5
                elif self.previous_ball_owned_team == 1 and current_obs['ball_owned_team'] == 0:
                    # Higher reward for regaining possession from the opponent
                    components["possession_change_reward"][idx] = 1.0

            # Update reward
            reward[idx] += components["possession_change_reward"][idx]

        # Update previous ball ownership state
        self.previous_ball_owned_team = current_obs['ball_owned_team']

        return reward, components

    def step(self, action):
        """
        Step function to apply actions, process observations, and extract rewards.
        """
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
        return observation, reward, done, info
