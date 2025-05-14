import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards to enhance spatial awareness and positional adjustments for defensive strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}  # Save relevant information if needed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Load relevant information if needed
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None or not observation:
            return reward, {}

        components = {"base_score_reward": reward.copy()}

        # Assuming there is only one agent in the environment for this subtask
        o = observation[0]  # Focus on the single agent's observation

        components['defensive_positioning_reward'] = [0.0]

        # Reward for maintaining optimal distances from opponents
        own_team = o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']
        opponent_team = o['left_team'] if o['ball_owned_team'] == 1 else o['right_team']

        # Calculate the average distance to all opponents
        optimal_distance = 0.05  # Define an optimal distance to keep from opponents
        distances = [np.linalg.norm(own_pos - opp_pos) for own_pos in own_team for opp_pos in opponent_team]
        avg_distance = np.mean(distances)
        if avg_distance > optimal_distance:
            components['defensive_positioning_reward'][0] = 0.1 * (avg_distance - optimal_distance)

        # Aggregate rewards
        total_reward = reward[0] + components['defensive_positioning_reward'][0]

        return [total_reward], components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
