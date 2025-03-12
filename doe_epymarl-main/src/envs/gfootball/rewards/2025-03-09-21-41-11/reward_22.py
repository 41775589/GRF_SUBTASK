import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper for offensive strategy development in FootballEnv."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._pass_accuracy_reward = 0.1
        self._dribble_success_reward = 0.2
        self._accurate_shot_reward = 0.3
        self._previous_ball_owned_team = None
        self._num_actions_taken = 0
        self.max_dribble_steps = 5  # Reward dribbling for at most these many continuous steps

    def reset(self, **kwargs):
        self._previous_ball_owned_team = None
        self._num_actions_taken = 0
        return self.env.reset(**kwargs)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
        
        components = {
            "base_score_reward": reward.copy(),
            "pass_accuracy_reward": 0.0,
            "dribble_success_reward": 0.0,
            "accurate_shot_reward": 0.0
        }

        ball_owned_team = observation['ball_owned_team']

        # Reward for maintaining possession and making passes
        if self._previous_ball_owned_team is not None:
            if ball_owned_team == self._previous_ball_owned_team:
                components["pass_accuracy_reward"] = self._pass_accuracy_reward
                reward += components["pass_accuracy_reward"]
        
        # Check dribble success:
        # Assuming a function is_dribbling_success() that uses 'sticky_actions' to determine dribbling success.
        if self._num_actions_taken <= self.max_dribble_steps:
            if observation['sticky_actions'][9]:  # assuming dribble action index 9
                components["dribble_success_reward"] = self._dribble_success_reward
                reward += components["dribble_success_reward"]

        # Reward for accurate shooting:
        # Assuming a function is_accurate_shot() that determines if a shot was accurate.
        if observation['game_mode'] == 6:  # Assuming game_mode 6 relates to a shot attempt
            components["accurate_shot_reward"] = self._accurate_shot_reward
            reward += components["accurate_shot_reward"]

        self._previous_ball_owned_team = ball_owned_team
        self._num_actions_taken += 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Add component values to info dict for transparency and debugging
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
