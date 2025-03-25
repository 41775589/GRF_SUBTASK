import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering long passes in football simulation."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.previous_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_threshold = 0.4  # Considered a long pass if the ball travels at least this distance
        self.pass_accuracy_bonus = 0.1   # Reward bonus for accurate long pass

    def reset(self):
        """Reset environment and variables."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of the environment and the wrapper."""
        to_pickle['previous_ball_position'] = self.previous_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the environment and the wrapper."""
        from_pickle = self.env.set_state(state)
        self.previous_ball_position = from_pickle['previous_ball_position']
        return from_pickle

    def reward(self, reward):
        """Modify reward based on the precision and distance of passes."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None or self.previous_ball_position is None:
            return reward

        # Determine the distance the ball traveled
        current_ball_position = observation['ball'][:2]
        distance_traveled = np.linalg.norm(current_ball_position - self.previous_ball_position)

        # Check if the pass was a long pass
        if distance_traveled >= self.long_pass_threshold:
            reward += self.pass_accuracy_bonus
        
        # Updating previous ball position for next reward calculation
        self.previous_ball_position = current_ball_position

        return reward, components

    def step(self, action):
        """Take an action and modify the game's dynamics."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Logging the reward components for debugging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
