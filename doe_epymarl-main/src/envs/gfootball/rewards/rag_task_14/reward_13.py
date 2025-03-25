import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for the 'sweeper' role aimed at ball clearing, tackling, and fast recoveries."""

    def __init__(self, env):
        super().__init__(env)
        self.previous_ball_position = np.zeros(3)
        self.steps_since_last_tackle = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # For recording sticky actions
        self.ball_clearance_zone = 0.75  # The x-coordinate limit dividing the defensive zone

    def reset(self):
        """Resets the wrapper states alongside the environment."""
        self.previous_ball_position = np.zeros(3)
        self.steps_since_last_tackle = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store the custom state when pickling."""
        to_pickle['previous_ball_position'] = self.previous_ball_position
        to_pickle['steps_since_last_tackle'] = self.steps_since_last_tackle
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the custom state upon unpickling."""
        from_pickle = self.env.set_state(state)
        self.previous_ball_position = from_pickle['previous_ball_position']
        self.steps_since_last_tackle = from_pickle['steps_since_last_tackle']
        return from_pickle

    def reward(self, reward):
        """Custom reward logic to motivate the sweeper role functionality."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for clearing the ball from the defensive zone
            if o['ball'][0] < self.ball_clearance_zone and np.linalg.norm(o['ball'] - self.previous_ball_position) > 0:
                components["clearance_reward"][rew_index] = 0.1
                reward[rew_index] += components["clearance_reward"][rew_index]

            # Reward for completing a tackle
            if self.steps_since_last_tackle < 50:  # Assuming a tackle would prevent a goal for at least 50 steps
                components["tackle_reward"][rew_index] = 0.2
                reward[rew_index] += components["tackle_reward"][rew_index]

            self.previous_ball_position = o['ball']
            self.steps_since_last_tackle += 1

        return reward, components

    def step(self, action):
        """Step function that processes the action and observation through the reward wrapper."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
