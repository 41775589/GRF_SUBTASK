import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper focused on the role of a 'sweeper'. It encourages the agent to:
    1) Clear balls from the defensive zone.
    2) Perform critical last-man tackles.
    3) Quickly recover and cover positions when needed.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_reward = 1.0
        self.tackle_reward = 1.0
        self.cover_position_reward = 0.5

    def reset(self):
        """Reset the sticky actions counter and other necessary states."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Retrieve the state for serialization."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from deserialization."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward function to emphasize sweeper roles.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": 0.0,
                      "tackle_reward": 0.0,
                      "cover_position_reward": 0.0}

        # Check for clearing balls from defensive zone
        if 'ball' in observation and observation['ball'][0] < -0.5:  # Ball in defensive half
            components['clearance_reward'] = self.clearance_reward

        # Check for tackles in critical last-man positions
        if 'ball_owned_team' in observation and observation['ball_owned_team'] == 1:  # Ball owned by opponent
            if 'ball_owned_player' in observation:
                distance_to_goal = 1 + observation['ball'][0]  # x position of the ball, assuming goal at x=-1
                if distance_to_goal < 0.2:
                    components['tackle_reward'] = self.tackle_reward

        # Reward for quickly covering open positions (simplified as moving towards the ball)
        if 'right_team' in observation and 'ball' in observation:
            player_pos = observation['right_team'][observation['active']]
            ball_pos = observation['ball'][:2]  # Ignore z-axis
            if np.linalg.norm(player_pos - ball_pos) < 0.1:
                components['cover_position_reward'] = self.cover_position_reward

        # Calculate the total reward
        total_additional_reward = sum(components.values()) - reward[0]
        reward[0] += total_additional_reward

        return reward, components

    def step(self, action):
        """Compute the environment's step and append new calculated rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, act in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = act
        return observation, reward, done, info
