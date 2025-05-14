import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a specific reward for training in shooting techniques,
    incentivizing shots from various positions under different game scenarios.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the rewards for different shooting conditions
        self.shot_reward = 1.0  # Reward for scoring a goal
        self.distance_reward = 0.2  # Reward for shooting from distance 

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        # Initialize reward components
        components = {
            "base_score_reward": reward.copy(),
            "shot_rewards": [0.0],
            "distance_rewards": [0.0]
        }
        
        if 'score' in observation:
            # Add a base reward if a goal is scored
            if observation['score'][0] > 0:
                components["shot_rewards"][0] = self.shot_reward
                reward[0] += components["shot_rewards"][0]

            # Calculate the distance from the goal and provide rewards for long distance shots
            ball_pos = observation['ball'][:2]  # Get x, y coordinates
            goal_pos = [1.0, 0.0]  # Assuming shooting towards right goal which is at x = 1.0
            distance = np.linalg.norm(np.array(ball_pos) - np.array(goal_pos))
            if distance > 0.5:  # Threshold for considering as a long-distance shot
                components["distance_rewards"][0] = self.distance_reward
                reward[0] += components["distance_rewards"][0]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Add detailed reward information in the output dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions counter
        if 'sticky_actions' in observation:
            for i, action_active in enumerate(observation['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
                else:
                    self.sticky_actions_counter[i] = 0
            for i in range(10):
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
