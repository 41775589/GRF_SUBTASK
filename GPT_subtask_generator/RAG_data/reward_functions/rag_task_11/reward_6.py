import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to provide specialized rewards to reinforce offensive capabilities such as fast-paced maneuvers and precision finishing."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.offensive_zone_reward = 0.05
        self.precision_finish_reward = 1.0
        self.speed_bonus_base = 0.02
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        # Initialize reward components dictionary and base score reward
        components = {"base_score_reward": reward.copy(),
                      "offensive_zone_reward": [0.0] * len(reward),
                      "precision_finish_reward": [0.0] * len(reward),
                      "speed_bonus": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for index, agent_obs in enumerate(observation):
            # Offensive zone reward increment when the player is close to the opponent's goal
            if agent_obs['ball'][0] > 0.5:  # Assuming ball position x > 0.5 is close to the opponent goal
                components["offensive_zone_reward"][index] += self.offensive_zone_reward
                reward[index] += components["offensive_zone_reward"][index]

            # Precision finishing reward when scoring
            if agent_obs['score'][0] > agent_obs['score'][1]:  # Check if left team's score is greater than right's
                components["precision_finish_reward"][index] = self.precision_finish_reward
                reward[index] += components["precision_finish_reward"][index]

            # Speed bonus based on the pace at which the player moves forward with the ball towards the opponent's goal
            if agent_obs['ball_owned_team'] == 0 and np.linalg.norm(agent_obs['ball_direction'][:2]) > 0.1:
                # Assumes significant ball movement is a fast-paced maneuver
                components["speed_bonus"][index] = self.speed_bonus_base * np.linalg.norm(agent_obs['ball_direction'][:2])
                reward[index] += components["speed_bonus"][index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
