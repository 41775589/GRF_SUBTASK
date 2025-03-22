import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focused on training agents to perfect sliding tackles in high-pressure situations."""

    def __init__(self, env):
        super().__init__(env)
        # Parameters related to checking successful sliding tackles
        self.sliding_tackle_success_reward = 1.0
        self.sliding_tackle_attempt_penalty = -0.1
        self.high_pressure_threshold = 0.2  # Proximity to opposing players defining high pressure
        self.sliding_action_index = 12  # Assuming sliding_action is at index 12

    def reset(self):
        """Reset the environment and wrapper states."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Collect internal state, no additional state needed for our wrapper."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set internal state, no additional state needed for our wrapper."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Customize reward: higher rewards for successful tackles under high pressure."""
        observation = self.env.unwrapped.observation()  # Provides access to the low-level observation
        if observation is None:
            return reward, {}

        modified_reward = reward.copy()
        components = {"base_score_reward": reward.copy(), "tackle_reward": 0.0}

        # Check if sliding action was taken
        if observation['sticky_actions'][self.sliding_action_index] == 1:
            ball_position = np.array(observation['ball'][:2])  # Take only x, y coordinates
            player_position = observation['right_team'][observation['active']]

            # Calculate the distance to the ball
            distance_to_ball = np.linalg.norm(ball_position - player_position)
            components['tackle_attempt'] = self.sliding_tackle_attempt_penalty

            # Detect high-pressure situation
            for opponent_position in observation['left_team']:
                if np.linalg.norm(player_position - opponent_position) <= self.high_pressure_threshold:
                    components['tackle_attempt'] = 0.0  # Cancel penalty in high pressure
                    if distance_to_ball < self.high_pressure_threshold:
                        # Reward for successful tackle under high pressure
                        components['tackle_reward'] = self.sliding_tackle_success_reward
                    break

        # Calculate the reward for sliding under pressure
        final_reward = modified_reward + components['tackle_reward'] + components['tackle_attempt']
        return final_reward, components

    def step(self, action):
        """Override the step to incorporate customized reward handling."""
        observation, reward, done, info = self.env.step(action)
        # Modify the reward
        reward, components = self.reward(reward)
        # Store the modified reward and components for debugging
        info["final_reward"] = reward
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
