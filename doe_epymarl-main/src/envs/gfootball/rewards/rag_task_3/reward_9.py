import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dedicated reward based on shooting accuracy and power."""

    def __init__(self, env):
        super().__init__(env)
        # Stickiness of actions to track continuous actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Coefficients for rewards
        self.accuracy_coefficient = 0.5
        self.power_coefficient = 0.5
        self.scenario_pressure_coefficient = 1.0

    def reset(self):
        # Reset sticky actions counter on reset
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Include this wrapper's state
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist(),
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Load this wrapper's state
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        # Processed observation from the environment
        observation = self.env.unwrapped.observation()

        components = {"base_score_reward": reward.copy(), "accuracy_reward": [0.0], "power_reward": [0.0]}

        if observation is None:
            return reward, components

        for player in observation:
            # Reward based on distance to goal and current ball velocity
            ball_position = player['ball']
            ball_direction = player['ball_direction']
            shot_power = np.linalg.norm(ball_direction[:2])

            # Goal positions are approximately x = 1 (right goal) or x = -1 (left goal)
            goal_position = np.array([1, 0]) if ball_position[0] < 0 else np.array([-1, 0])
            distance_to_goal = np.linalg.norm(ball_position[:2] - goal_position)

            # Calculate accuracy and power components
            accuracy_reward = max(0, self.accuracy_coefficient * (1 - distance_to_goal))
            power_reward = self.power_coefficient * shot_power

            # Check if the player is in a shooting situation under pressure
            if player['game_mode'] in [2, 3, 4]:  # Pressure scenarios: FreeKick, Corner, or GoalKick
                total_reward = accuracy_reward + power_reward
                components['accuracy_reward'][0] += accuracy_reward * self.scenario_pressure_coefficient
                components['power_reward'][0] += power_reward * self.scenario_pressure_coefficient
                reward[0] += total_reward * self.scenario_pressure_coefficient
            else:
                components['accuracy_reward'][0] += accuracy_reward
                components['power_reward'][0] += power_reward
                reward[0] += accuracy_reward + power_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
