import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A custom reward wrapper for a midfield/defense training task in a soccer simulation."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize tracking for transitions and control effectiveness.
        self.ball_control_count = 0
        self.defensive_actions = 0
        self.passing_accuracy = 0

    def reset(self):
        self.ball_control_count = 0
        self.defensive_actions = 0
        self.passing_accuracy = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        o = observation[0]
        modified_reward = [reward[0]]  # Ensure reward is in list format for consistency.
        components = {
            "base_score_reward": [reward[0]],
            "control_reward": [0.0],
            "defense_reward": [0.0],
            "pass_reward": [0.0]
        }

        # Reward increases for effective ball control and field progression.
        if o['ball_owned_player'] == o['active']:
            self.ball_control_count += 1
            components["control_reward"][0] = 0.05 * self.ball_control_count
        
        # Reward for defensive plays when game_mode indicates a defensive situation.
        if o['game_mode'] == 2:  # Assuming mode 2 is a defensive game mode
            self.defensive_actions += 1
            components["defense_reward"][0] = 0.1 * self.defensive_actions
        
        # Reward for successful passes.
        if o['game_mode'] == 4 or o['game_mode'] == 5:  # Assuming these modes are pass modes
            self.passing_accuracy += 1
            components["pass_reward"][0] = 0.2 * self.passing_accuracy

        # Sum all component rewards with the base environment reward.
        total_reward = sum(modified_reward + [value[0] for value in components.values()])
        modified_reward = [total_reward]  # Update the reward list to reflect total reward.

        return modified_reward, components

    def step(self, action):
        """Executes a step in the environment and adjusts the reward returned."""
        obs, reward, done, info = self.env.step(action)
        modified_reward, components = self.reward(reward)
        
        # Add final adjusted reward and reward components to the info dictionary.
        info['final_reward'] = sum(modified_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return obs, modified_reward, done, info
