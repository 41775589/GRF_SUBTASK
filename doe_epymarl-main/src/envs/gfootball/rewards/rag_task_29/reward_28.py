import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a goal-oriented reward focused on shot precision and power adjustment in tight spaces."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the target area for precision training
        # Assuming middle of opponent's goal area as crucial points to hit
        # Coordinates may be environment specific and need adjustment
        self.target_points = [(1.0, 0.0)]  # Center of the opponent's goal
        self.precision_threshold = 0.1  # Defines the closeness to the target point to get reward

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        """Calculate rewards based on shot precision close to the goal."""
        observation = self.env.unwrapped.observation()
        
        # Setup reward components for each agent
        components = {"base_score_reward": reward.copy(),
                      "precision_reward": [0.0, 0.0]}
        
        for i, o in enumerate(observation):
            if o['game_mode'] in (6,):  # Checking if this is a penalty kick
                ball_position = o['ball'][:2]  # Usually consists X, Y (and Z if present)

                # Reward for precision
                for target in self.target_points:
                    distance = np.linalg.norm(np.array(ball_position) - np.array(target))
                    if distance <= self.precision_threshold:
                        components["precision_reward"][i] = 1.0  # Assign a fixed reward for high precision
                        break

                # Combine base reward with extra precision reward
                reward[i] += components["precision_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
