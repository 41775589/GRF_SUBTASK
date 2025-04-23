import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for shooting training."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initialize counter for sticky actions

    def reset(self):
        """Reset for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the internal state of the wrapper and environment."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the internal state of the wrapper and environment."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on the agent's position related to goal and ball control."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'shot_precision_reward': [0.0] * len(reward)}
        goal_y_range = [-0.044, 0.044]  # Y-range for goal
        potent_shot_dist = 0.2  # define distance from the goal to consider for shooting context

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            own_goal_pos = -1 if o['ball_owned_team'] == 1 else 1  # Position of own goal on x-axis

            ball_pos = o['ball'][:2]  # x, y position of the ball
            player_pos = o['left_team'][o['active']][:2] if o['ball_owned_team'] == 0 else o['right_team'][o['active']][:2]

            goal_distance = np.sqrt((ball_pos[0] - own_goal_pos)**2 + (ball_pos[1])**2)
            angle_toward_goal = np.arctan2(own_goal_pos - ball_pos[0], ball_pos[1])

            # Define rewards for kicking accuracy and avoiding the keeper assuming keeper at the center of goal scope.
            if goal_distance < potent_shot_dist:
                angle_penalty = abs(angle_toward_goal) / np.pi  # Normalized angle penalty
                distance_bonus = (potent_shot_dist - goal_distance) / potent_shot_dist  # Closer is better
                components['shot_precision_reward'][rew_index] = (0.5 * distance_bonus + 0.5 * (1 - angle_penalty)) * 2.0

            reward[rew_index] += components['shot_precision_reward'][rew_index] * 5.0  # Scale the shooting reward for significant impact

        return reward, components

    def step(self, action):
        """Override step to include reward processing."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
