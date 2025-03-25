import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on defensive strategies and ball control."""

    def __init__(self, env):
        super().__init__(env)
        self.num_defensive_zones = 5  # Define the number of zones for defensive coordination
        self.zone_reward = 0.05  # Reward for maintaining positioning in a defensive zone
        self.ball_control_reward = 0.1  # Reward for controlling the ball in defensive area
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Retrieve state for saving."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set state from loaded."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Custom reward function enhancing defensive collaboration and ball control."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_position_reward": [0.0] * len(reward),
                      "ball_control_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Ensure observation is valid.
            if o is None:
                continue
            
            # Calculate position based defensive rewards
            if 'left_team' in o and 'ball' in o and o['ball_owned_team'] == 0:
                ball_zone = int((o['ball'][0] + 1) * self.num_defensive_zones / 2)
                player_zone = int((o['left_team'][o['active']][0] + 1) * self.num_defensive_zones / 2)
                if ball_zone == player_zone:
                    # The player is in the same zone as the ball, good defensive shadowing
                    components["defensive_position_reward"][rew_index] = self.zone_reward
                    reward[rew_index] += components["defensive_position_reward"][rew_index]
            
            # Calculate ball control rewards in defensive areas
            if o['ball_owned_team'] == 0 and o['ball'][0] < -0.5:  # Ball in defensive half
                if o['active'] == o['ball_owned_player']:
                    components["ball_control_reward"][rew_index] = self.ball_control_reward
                    reward[rew_index] += components["ball_control_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Apply action, step the environment, and apply the reward wrapper."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
