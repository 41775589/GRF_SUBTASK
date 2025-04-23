import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper focusing on wingers' crossing and sprinting training with explicit rewards for crossing accuracy and high-speed dribbling."""

    def __init__(self, env):
        super().__init__(env)
        self.checkpoint_reached = False  # Initialize to indicate crossing achievement

    def reset(self):
        """Reset the environment and clear checkpoints."""
        self.checkpoint_reached = False
        return self.env.reset()

    def reward(self, reward):
        """Modify the reward based on winger's performance on crossing accuracy and high-speed dribbling."""
        observation = self.env.unwrapped.observations()
        reward_components = {
            "base_score_reward": reward.copy(),
            "crossing_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward)
        }

        for idx, obs in enumerate(observation):
            # Conditions for rewarding crossing:
            # High ball possession in the side of the field (representing a cross)
            # This part simplifies how one might determine a successful cross
            if obs['ball_owned_team'] == 1 and abs(obs['ball'][1]) >= 0.30:
                reward_components["crossing_reward"][idx] = 0.2
                reward[idx] += reward_components["crossing_reward"][idx]

            # Conditions for rewarding dribbling:
            # Player has the ball and is near the side of the field, engaging in a sprint
            if (obs['ball_owned_team'] == 1 and obs['stick_actions'][8] == 1 and
                    abs(obs['ball'][0]) > 0.70):
                # Encourage sprinting down the wing
                reward_components["dribbling_reward"][idx] = 0.1
                reward[idx] += reward_components["dribbling_reward"][idx]
        
        return reward, reward_components

    def step(self, action):
        """Take a step using the given action, modify the reward, and return the new state, reward, done, and info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
