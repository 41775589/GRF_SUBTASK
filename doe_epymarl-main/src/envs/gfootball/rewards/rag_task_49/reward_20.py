import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful shooting from central field positions with high accuracy and power."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_accuracy_threshold = 0.1  # Distance threshold to the goal center for considering accurate
        self.shooting_power_threshold = 0.5  # Minimal power threshold for the shot
        self.central_zone_range = (-0.25, 0.25)  # X range defining the central zone of the field
        self.shooting_success_reward = 1.0  # Reward given for a successful shot

    def reward(self, reward):
        components = {'base_score_reward': reward.copy(), 'shooting_reward': [0.0] * len(reward)}
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Check if the shot is taken from the central field zone
            if self.central_zone_range[0] < o['ball'][0] < self.central_zone_range[1]:
                # Calculate the distance from the ball to the goal center
                goal_center_distance = abs(o['ball'][1])  # Goal center is at y=0
                # Check if the ball is heading towards the goal center with sufficiently high power
                if goal_center_distance <= self.shooting_accuracy_threshold and np.linalg.norm(o['ball_direction'][:2]) >= self.shooting_power_threshold:
                    # Add a high reward for accurate and powerful shots from the central zone
                    components['shooting_reward'][i] = self.shooting_success_reward
                    reward[i] += components['shooting_reward'][i]

        return reward, components

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
