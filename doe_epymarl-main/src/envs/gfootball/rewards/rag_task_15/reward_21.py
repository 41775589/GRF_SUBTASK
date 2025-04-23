import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering long passes."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the checkpoints for distance in normalized pitch coordinates [0-1] ([0, 0] is left goal, [1, 1] is right goal)
        self.distance_thresholds = np.linspace(0.3, 0.9, 7)  # Define parameters for what counts as a "long" pass
        self.long_pass_reward = 0.05      # Reward increment for each pass over a distance threshold
        self._last_ball_position = None   # To store the last position of the ball

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._last_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._last_ball_position = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "long_pass_reward": [0.0] * len(reward)}
        
        # Ensure the observation is available
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_ball_position = o['ball'][:2]  # We consider only X, Y plane

            # Store initial ball position at the start of the episode
            if self._last_ball_position is None:
                self._last_ball_position = current_ball_position

            # Compute the distance the ball has traveled since the last step
            ball_travel_distance = np.sqrt(np.sum((np.array(current_ball_position) - np.array(self._last_ball_position))**2))

            # Check if the ball was passed a significant distance
            passed_thresholds = np.sum(ball_travel_distance > self.distance_thresholds)
            if passed_thresholds > 0:
                additional_reward = passed_thresholds * self.long_pass_reward
                components["long_pass_reward"][rew_index] += additional_reward
                reward[rew_index] += additional_reward
            
            # Update the last ball position
            self._last_ball_position = current_ball_position
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update action counters based on current sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
