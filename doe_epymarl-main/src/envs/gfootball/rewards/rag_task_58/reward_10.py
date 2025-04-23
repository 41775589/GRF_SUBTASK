import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward to encourage defensive coordination and efficient transitions."""

    def __init__(self, env):
        super().__init__(env)
        self._max_dist_to_goal = 1.0
        self._distance_reward = 0.05  # Reward increment for decreasing the distance by a set amount
        self._possession_reward = 0.1  # Reward for maintaining possession of the ball
        self._transition_reward = 0.2  # Reward for successful transition from defense to attack
        self._last_ball_owner = None
        self._last_attack_position = None
        self._last_defense_position = None

    def reset(self):
        self._last_ball_owner = None
        self._last_attack_position = None
        self._last_defense_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'last_ball_owner': self._last_ball_owner,
            'last_attack_position': self._last_attack_position,
            'last_defense_position': self._last_defense_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._last_ball_owner = from_pickle['CheckpointRewardWrapper']['last_ball_owner']
        self._last_attack_position = from_pickle['CheckpointRewardWrapper']['last_attack_position']
        self._last_defense_position = from_pickle['CheckpointRewardWrapper']['last_defense_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward),
                      "distance_reward": [0.0] * len(reward)}

        for idx, o in enumerate(observation):
            # Calculating possession maintenance
            if o['ball_owned_team'] == o['active_team'] and (self._last_ball_owner == o['active_team']):
                components["possession_reward"][idx] += self._possession_reward

            # Calculating defense to attack transition
            if o['ball_owned_team'] == o['active_team']:
                if self._last_defense_position is not None and o['ball'][0] > self._last_defense_position[0]:
                    components["transition_reward"][idx] += self._transition_reward
                    self._last_attack_position = o['ball'].copy()

            # Calculate distance reward based on position closer to opponent's goal
            normalized_dist = np.linalg.norm([o['ball'][0] - 1, o['ball'][1]]) / self._max_dist_to_goal
            if self._last_ball_owner == o['active_team'] and self._last_attack_position is not None:
                previous_normalized_dist = np.linalg.norm([self._last_attack_position[0] - 1, self._last_attack_position[1]]) / self._max_dist_to_goal
                if normalized_dist < previous_normalized_dist:
                    components["distance_reward"][idx] += self._distance_reward

            # Update states of ball ownership and ball position if the current team owns the ball
            if o['ball_owned_team'] == o['active_team']:
                self._last_ball_owner = o['active_team']
                self._last_defense_position = o['ball'].copy()

            reward[idx] += components["possession_reward"][idx] + components["transition_reward"][idx] + components["distance_reward"][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
