import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward function to train an agent functioning as a midfielder/advance defender."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_accuracy_bonus = 1.0
        self.dribble_effectiveness_bonus = 1.0
        self.sprint_utility_bonus = 0.5
        self.ball_possession_bonus = 0.3

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['game_state'] = self.env.get_state()
        return to_pickle

    def set_state(self, state):
        _ = self.env.set_state(state['game_state'])
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        o = observation[0]  # Assuming single agent in control

        components = {"base_score_reward": reward[0],
                      "pass_accuracy_bonus": 0,
                      "dribble_effectiveness_bonus": 0,
                      "sprint_utility_bonus": 0,
                      "ball_possession_bonus": 0}

        # Check ball possession and add bonus for holding the ball
        if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
            components['ball_possession_bonus'] = self.ball_possession_bonus
            reward[0] += components['ball_possession_bonus']

        # Reward for successful passes while playing as midfielder
        if o['game_mode'] in (1, 2):  # Game modes that imply passing
            components['pass_accuracy_bonus'] = self.pass_accuracy_bonus
            reward[0] += components['pass_accuracy_bonus']

        # Dribbling improves ball control, thus bonus if successfully avoiding opponents
        if o['sticky_actions'][7] == 1:  # Assuming index 7 is the dribble action
            components['dribble_effectiveness_bonus'] = self.dribble_effectiveness_bonus
            reward[0] += components['dribble_effectiveness_bonus']

        # Sprinting and stopping sprints effectively to reposition
        if o['sticky_actions'][8] == 1:  # Assuming index 8 is sprint
            components['sprint_utility_bonus'] = self.sprint_utility_bonus
            reward[0] += components['sprint_utility_bonus']

        return [reward[0]], components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        # Add individual components as well
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
