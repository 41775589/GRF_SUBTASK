import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for optimal positioning for shooting under high pressure."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for reward calculations
        self.positioning_threshold = 0.15  # Optimal distance range from the goal
        self.pressure_threshold = 0.2  # Pressure by opponent defining high-pressure situations
        self.optimal_shooting_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['RewardWrapper'] = {"sticky_actions_counter": self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['RewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "optimal_shooting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            if obs['game_mode'] != 0:  # Skip checks during game interruptions
                continue

            # Get player positions and ball positions
            ball_position = obs['ball'][:2]  # Ignore z-component for distances
            player_pos = obs['left_team'][obs['active']]
            opponents = obs['right_team']

            # Calculate distance to the goal and to nearest opponent
            distance_to_goal = np.linalg.norm(ball_position - np.array([1, 0]))
            nearest_opponent_distance = np.min([np.linalg.norm(player_pos - op_pos) for op_pos in opponents])

            # High pressure situation and good positioning for a shot
            if nearest_opponent_distance < self.pressure_threshold and distance_to_goal <= self.positioning_threshold:
                components['optimal_shooting_reward'][rew_index] = self.optimal_shooting_reward
                reward[rew_index] += components['optimal_shooting_reward'][rew_index]

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
