import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modulates rewards for spatial awareness and positioning for defensive strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize positional reward configuration
        self.goal_line_x = 1.0 # x-coordinate of the opponent's goal line
        self.defensive_line_reward_multiplier = 0.05 # Reward multiplier for positioning
        self.opponent_half_multiplier = 0.1 # Extra multiplier when in opponent's half

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
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(), "defensive_reward": [0.0]}
        player_pos = observation[0]['right_team'][observation[0]['active']]
        
        # Calculate the x-distance from own goal (normalize with field length, [-1, 1])
        distance_from_own_goal = player_pos[0] + 1
        
        # Basic midfield positioning reward when player is positioned in their own half
        if player_pos[0] < 0:
            components["defensive_reward"][0] = distance_from_own_goal * self.defensive_line_reward_multiplier
        
        # Higher reward for positioning in the opponent's half
        if player_pos[0] > 0:
            components["defensive_reward"][0] += (self.goal_line_x - player_pos[0]) * self.opponent_half_multiplier
        
        total_reward = reward[0] + components["defensive_reward"][0]
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
