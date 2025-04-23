import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward focusing on improving offensive skills
    such as passing, shooting, and dribbling to create scoring opportunities.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.05
        self.shot_reward = 0.1
        self.dribble_reward = 0.03
        self.sprint_reward = 0.02

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        # Initializing components dictionary
        components["pass_reward"] = [0.0] * len(reward)
        components["shot_reward"] = [0.0] * len(reward)
        components["dribble_reward"] = [0.0] * len(reward)
        components["sprint_reward"] = [0.0] * len(reward)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check for pass actions
            # sticky_actions[6] is for Short Pass, sticky_actions[8] is for Long Pass
            if o['sticky_actions'][6] == 1 or o['sticky_actions'][8] == 1:
                components["pass_reward"][rew_index] = self.pass_reward
                reward[rew_index] += components["pass_reward"][rew_index]

            # Check for shot actions
            # sticky_actions[9] represents action Shot
            if o['sticky_actions'][9] == 1:
                components["shot_reward"][rew_index] = self.shot_reward
                reward[rew_index] += components["shot_reward"][rew_index]

            # Check for dribble actions
            # sticky_actions[7] represents action Dribble
            if o['sticky_actions'][7] == 1:
                components["dribble_reward"][rew_index] = self.dribble_reward
                reward[rew_index] += components["dribble_reward"][rew_index]

            # Check for sprint actions
            # sticky_actions[3] represents action Sprint
            if o['sticky_actions'][3] == 1:
                components["sprint_reward"][rew_index] = self.sprint_reward
                reward[rew_index] += components["sprint_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
