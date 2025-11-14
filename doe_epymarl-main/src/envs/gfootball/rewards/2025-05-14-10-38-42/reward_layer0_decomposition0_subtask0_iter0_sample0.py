import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effective offensive play, focusing on shot accuracy and dynamic movement strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the thresholds for proximity to the goal for shot accuracy
        self.proximity_to_goal_threshold = 0.1  # e.g. 10% distance to the goal
        self.shooting_reward_coefficient = 2.0  # Reward multiplier for shots near goal
        self.movement_reward_coefficient = 0.5  # Reward multiplier for effective movement

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'shooting_reward': [0.0] * len(reward),
                      'movement_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            components["base_score_reward"][rew_index] = reward[rew_index]
            
            # Check if shot is made near the goal
            goal_distance = np.linalg.norm([o['ball'][0] - 1, o['ball'][1]])  # distance to right goal
            if o['ball_owned_player'] == o['active'] and goal_distance < self.proximity_to_goal_threshold:
                components["shooting_reward"][rew_index] = self.shooting_reward_coefficient * reward[rew_index]
            
            # Add movement effectiveness reward based on the action effectiveness towards goal movement
            # Approximate dynamic movement: direction towards the opponent's half improving
            if o['ball_owned_team'] == 1:  # assuming the right team is the opponent
                if o['ball_direction'][0] > 0:  # ball moving towards the right side (opponent goal)
                    components["movement_reward"][rew_index] = self.movement_reward_coefficient * reward[rew_index]

            reward[rew_index] += (components["shooting_reward"][rew_index] + components["movement_reward"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
