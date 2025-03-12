import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward for mastering sliding tackles when in defense."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Sliding tackle success reward multiplier
        self.sliding_success_reward = 5.0
        # Penalty for unsuccessful sliding tackle
        self.sliding_failure_penalty = -1.0
        # Track the last position to determine movement
        self.last_ball_position = np.array([0.0, 0.0, 0.0])

    def reset(self):
        """ Reset environment and save initial state of the ball."""
        obs = self.env.reset()
        self.last_ball_position = obs['ball']  # Reset the last known ball position
        return obs

    def reward(self, reward):
        """ Modify reward based on defensive action effectiveness, mainly focusing on sliding tackles."""
        observation = self.env.unwrapped.observation()
        current_ball_position = observation['ball']
        ball_owned_team = observation['ball_owned_team']
        ball_displacement = np.linalg.norm(current_ball_position - self.last_ball_position)

        # Base reward component calculations
        components = {
            "base_score_reward": reward,
            "sliding_tackle_reward": 0.0
        }

        # Check if the ball is owned by the opponent
        if ball_owned_team == 1:  # Assuming the trained agent's team is 0
            # Active player tries a sliding tackle:
            if 'action_sliding' in observation and observation['action_sliding']:
                # If the ball displacement is minimal, it suggests a successful sliding tackle
                if ball_displacement < 0.01:  # Assuming small displacement suggests possession change
                    reward += self.sliding_success_reward
                    components["sliding_tackle_reward"] = self.sliding_success_reward
                else:
                    reward += self.sliding_failure_penalty
                    components["sliding_tackle_reward"] = self.sliding_failure_penalty

        # Update the last known position of the ball
        self.last_ball_position = current_ball_position

        return reward, components

    def step(self, action):
        """ Step through environment, modify reward, and record reward components."""
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info.update(components)  # Add detailed components to the info dictionary
        return obs, reward, done, info
