import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A gym wrapper to add a dense reward based on offensive skills focusing on
    passing, shooting, and dribbling to create scoring opportunities. It targets
    actions like Short Pass, Long Pass, Shot, Dribble, and Sprint.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define coefficients for each targeted sticky action
        self.short_pass_coeff = 0.05
        self.long_pass_coeff = 0.1
        self.shot_coeff = 0.2
        self.dribble_coeff = 0.075
        self.sprint_coeff = 0.05

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
                      "short_pass_reward": [0.0] * len(reward),
                      "long_pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            sticky_actions = obs['sticky_actions']

            if sticky_actions[7]:  # short_pass
                components["short_pass_reward"][rew_index] = self.short_pass_coeff
            if sticky_actions[9]:  # long_pass
                components["long_pass_reward"][rew_index] = self.long_pass_coeff
            if sticky_actions[8]:  # shot
                components["shot_reward"][rew_index] = self.shot_coeff
            if sticky_actions[1]:  # dribble
                components["dribble_reward"][rew_index] += self.dribble_coeff
            if sticky_actions[0]:  # sprint
                components["sprint_reward"][rew_index] = self.sprint_coeff

            # Sum the additional rewards
            reward[rew_index] += (components["short_pass_reward"][rew_index] +
                                  components["long_pass_reward"][rew_index] +
                                  components["shot_reward"][rew_index] +
                                  components["dribble_reward"][rew_index] +
                                  components["sprint_reward"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions counts after the step
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                if action == 1:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
