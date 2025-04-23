import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive maneuvers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define checkpoints for defensive positioning (proximity to our team's goal under threat)
        self.defensive_positions = np.linspace(-1, 0, 10)  # Positions divided along the x-axis of football field
        self.defensive_rewards = np.zeros_like(self.defensive_positions)
        self.position_reward_multiplier = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_rewards = np.zeros_like(self.defensive_positions)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_rewards'] = self.defensive_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_rewards = from_pickle.get('defensive_rewards', self.defensive_rewards)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'left_team' in o:
                # Calculate closest defensive checkpoint for each player
                player_positions = o['left_team'][:, 0]  # Get x-coordinates
                reward_contrib = 0
                for pos in player_positions:
                    # Check if player is in one of the defensive zones and the team does not own the ball
                    if o['ball_owned_team'] != 0:  # If ball is not owned by own team
                        diffs = np.abs(self.defensive_positions - pos)
                        closest_point_index = np.argmin(diffs)
                        if diffs[closest_point_index] < 0.1:  # Threshold for rewards for being around defensive zones
                            if self.defensive_rewards[closest_point_index] == 0:
                                reward_contrib += self.position_reward_multiplier
                                self.defensive_rewards[closest_point_index] = 1  # Ensure reward is only given once

                components["defensive_positioning"][rew_index] = reward_contrib
                reward[rew_index] += reward_contrib

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
