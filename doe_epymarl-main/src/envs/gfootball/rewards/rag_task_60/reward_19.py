import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on defensive positioning and quick transitions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_rewards = {}
        self._num_zones = 5  # Dividing each half of the pitch into 5 zones
        self._zone_reward = 0.05
        self._transition_bonus = 0.1
        self._previous_ball_position = None

    def reset(self, **kwargs):
        self._defensive_rewards = {}
        self._previous_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            # Check team's ball possession
            if o['ball_owned_team'] != o['active']:
                continue

            # Calculate distance to own goal (defensive positioning)
            own_goal_position = -1 if o['active'] == 0 else 1
            ball_distance_to_own_goal = abs(o['ball'][0] - own_goal_position)

            # Determine which zone the ball is in
            zone_index = int((ball_distance_to_own_goal / 2.0) * self._num_zones)
            if zone_index not in self._defensive_rewards:
                self._defensive_rewards[zone_index] = self._zone_reward
                components["defensive_positioning_reward"][i] += self._zone_reward

            # Reward for quick transitions between moving and not-moving
            if self._previous_ball_position is not None:
                movement = np.linalg.norm(o['ball'] - self._previous_ball_position)
                if movement > 0.1:  # Assumes significant movement
                    reward[i] += self._transition_bonus

            # Update previous position
            self._previous_ball_position = o['ball'].copy()

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = reward.sum()
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
