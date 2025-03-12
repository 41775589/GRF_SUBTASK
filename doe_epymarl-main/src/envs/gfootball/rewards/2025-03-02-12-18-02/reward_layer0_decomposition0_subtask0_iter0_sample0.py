import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper for the subtask of learning passing actions in a football game."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._reward_coefficient = 1.0  # Coefficient for reward adjustment
        self._passing_actions = ["Short Pass", "High Pass", "Long Pass"]  # List of passing actions to learn

    def reward(self, reward: list) -> tuple[list, dict[str, list]]:
        observations = self.env.unwrapped.observation()  # Get observations from the environment
        
        # Reward components
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": [0.0] * len(reward)  # Reward for executing passing actions
        }
        
        if observations is None:
            return reward, components
        
        for i, o in enumerate(observations):
            # Check if the action is one of the passing actions
            if o['action'] in self._passing_actions:
                components["passing_reward"][i] = 0.5  # Assign reward for executing passing actions
                reward[i] += self._reward_coefficient * components["passing_reward"][i]  # Add the adjusted reward
        
        return reward, components
