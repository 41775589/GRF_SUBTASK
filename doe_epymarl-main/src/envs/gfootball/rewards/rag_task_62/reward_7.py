import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on shooting angles and high-pressure play near the goal."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {"base_score_reward": reward.copy()}

        # Customize rewards based on the task
        for idx, o in enumerate(observation):
            # Assess the ball position relative to goal and controlled player
            goal_x = 1 if o['ball'][0] > 0 else -1
            player_x = o['right_team'][o['active']][0] if goal_x == 1 else o['left_team'][o['active']][0]

            # Encourage attacking gameplay: reward player getting closer to opposite goal with ball
            distance_to_goal = (1 - abs(o['ball'][0] - goal_x))
            components[f"attack_play_{idx}"] = distance_to_goal ** 2 * 0.1

            # High-pressure plays near the goal
            if abs(o['ball'][0] - goal_x) < 0.2 and o['ball_owned_team'] == (1 if goal_x == 1 else 0):
                components[f"high_pressure_{idx}"] = 0.3
            
            # Shooting angle fidelity
            if o['ball_owned_team'] == (1 if goal_x == 1 else 0) and o['ball'][1] ** 2 < 0.015:  # central vertical band near goal
                components[f"shooting_angle_{idx}"] = 0.5

            # Update rewards based on components
            total_additional_reward = sum(components[key][idx] for key in components if isinstance(components[key], list))
            reward[idx] += total_additional_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            if isinstance(value, list):
                info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
