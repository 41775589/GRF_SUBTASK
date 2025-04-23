import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for enhancing shot precision skills in tight spaces."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goal_distance_threshold = 0.2  # Ensures close range
        self.angle_threshold = 0.1  # Tolerance for angles to consider shot on target
        self.power_threshold = 0.1  # Tolerance for power adjustment

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
        components = {
            "base_score_reward": reward.copy(),
            "precision_reward": [0.0] * len(reward)
        }

        for i in range(len(reward)):
            o = observation[i]
            ball_position = o['ball'][0]  # Considering only x-coordinate for simplicity
            ball_direction = np.linalg.norm(o['ball_direction'][:2])
            ball_power = np.abs(o['ball_direction'][2])  # Consider vertical component for shot power

            # Calculate the bonuses for being in a tight spot near the goal and making a precise shot
            if abs(ball_position) > (1 - self.goal_distance_threshold):
                if ball_direction < self.angle_threshold and ball_power < self.power_threshold:
                    components['precision_reward'][i] = 1.0  # Give a high reward for precise shots
                reward[i] += components['precision_reward'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        observation = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for j, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{j}"] = action
        return observation, reward, done, info
