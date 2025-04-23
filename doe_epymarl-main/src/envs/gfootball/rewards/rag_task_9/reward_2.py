import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on offensive skills reinforcement in football scenarios.
       It particularly rewards passes, shots, dribbles, and sprints that lead towards scoring opportunities."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_bonus": [0.0] * len(reward),
                      "shot_bonus": [0.0] * len(reward),
                      "dribble_bonus": [0.0] * len(reward),
                      "sprint_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Checking if there was any sticky action for passing or dribbling
            if o['sticky_actions'][7] or o['sticky_actions'][1]:  # Long Pass or Short Pass
                components["pass_bonus"][rew_index] = 0.02
            if o['sticky_actions'][9]:  # Dribble
                components["dribble_bonus"][rew_index] = 0.01
            if o['sticky_actions'][8]:  # Sprint
                components["sprint_bonus"][rew_index] = 0.01

            # Check if shot was taken
            if o['game_mode'] == 6 and (o['ball_owned_team'] == o['left_team_roles'][rew_index] or 
                                         o['ball_owned_team'] == o['right_team_roles'][rew_index]):  # Game mode 6 is a shot
                components["shot_bonus"][rew_index] = 0.05

            reward[rew_index] += (components["pass_bonus"][rew_index] +
                                  components["shot_bonus"][rew_index] +
                                  components["dribble_bonus"][rew_index] +
                                  components["sprint_bonus"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
