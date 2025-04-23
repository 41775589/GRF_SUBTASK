import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that emphasizes maintaining strategic positioning and transitioning 
    between defensive and offensive plays.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_coefficient = 0.05
        self.transition_coefficient = 1.0

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
        observations = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "positioning_reward": [0.0] * len(reward)}

        if observations is None:
            return reward, components

        for i, obs in enumerate(observations):
            # Encourage strategic positioning
            positioning_score = self.evaluate_positioning_reward(obs)
            components["positioning_reward"][i] = positioning_score * self.position_coefficient
            reward[i] += components["positioning_reward"][i]

            # Encourage dynamic transitions between defense and offense
            if self.is_transitioning(obs):
                reward[i] += self.transition_coefficient

        return reward, components

    def evaluate_positioning_reward(self, observation):
        # Example simplistic logic: reward being near midfield when not possessing the ball
        if observation['ball_owned_team'] == 1 or observation['ball_owned_team'] == -1:
            player_x_position = observation['left_team'][observation['active']][0]  # Active player's x-coordinate
            return max(0, 1 - abs(player_x_position))  # Reward being near x = 0 (midfield)
        return 0

    def is_transitioning(self, observation):
        # Example logic: detect transition by big change in player's x-coordinate
        previous_x = observation['left_team_direction'][observation['active']][0]
        current_x = observation['left_team'][observation['active']][0]
        delta_x = abs(current_x - previous_x)
        return delta_x > 0.05  # Arbitrary threshold for significant horizontal movement

    def step(self, action):
        observations, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observations, reward, done, info
