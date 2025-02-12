import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward system based on various actions performed by the agent
    operating as a midfielder/advance defender. The rewards are tuned to emphasize efficient
    passing, defending, and maintaining possession under pressure.
    """

    def __init__(self, env):
        super().__init__(env)
        self.team_possession_count = 0
        self.successful_passes = 0
        self.defensive_actions = 0
        self.possession_under_pressure = 0

    def reset(self):
        self.team_possession_count = 0
        self.successful_passes = 0
        self.defensive_actions = 0
        self.possession_under_pressure = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle = self.env.get_state(to_pickle)
        to_pickle['team_possession_count'] = self.team_possession_count
        to_pickle['successful_passes'] = self.successful_passes
        to_pickle['defensive_actions'] = self.defensive_actions
        to_pickle['possession_under_pressure'] = self.possession_under_pressure
        return to_pickle

    def set_state(self, from_pickle):
        state = self.env.set_state(from_pickle)
        self.team_possession_count = from_pickle['team_possession_count']
        self.successful_passes = from_pickle['successful_passes']
        self.defensive_actions = from_pickle['defensive_actions']
        self.possession_under_pressure = from_pickle['possession_under_pressure']
        return state

    def reward(self, reward):
        """
        Customized reward function to foster learning specific sub-tasks:
        1. Promote possession when under pressure.
        2. Reward successful high and long passes.
        3. Reward tactical defensive positioning and actions.
        """
        observation = self.env.unwrapped.observation()[0]  # Assume single-agent scenario
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": 0.0,
            "defensive_reward": 0.0,
            "pressure_possession_reward": 0.0
        }

        # Check for successful passes in desirable situations
        if observation['ball_owned_team'] == 0 and observation['ball_owned_player'] == observation['active']:
            self.successful_passes += 1
            components['pass_reward'] = 0.2

        # Defensive actions reward: tackles, interceptions, effective marking
        if observation['game_mode'] in [3, 4]:  # Assuming these game modes are defensive
            self.defensive_actions += 1
            components['defensive_reward'] = 0.3

        # Possession under pressure
        if 'pressure' in observation and observation['pressure']:
            self.possession_under_pressure += 1
            components['pressure_possession_reward'] = 0.5

        # Calculate total modified reward
        total_reward = sum(reward) + sum(components.values())
        reward = [total_reward]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Record detailed reward components to track learning progress
        info["final_reward"] = reward[0]
        info.update({f"component_{k}": v for k, v in components.items()})
        
        return observation, reward[0], done, info
