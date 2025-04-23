import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward to encourage offensive skills such as passing, shooting, 
    and dribbling. It specifically rewards Short Pass, Long Pass, Shot, Dribble, and Sprint actions.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.1
        self.shot_reward = 0.2
        self.dribble_reward = 0.05
        self.sprint_reward = 0.03
        # Indices corresponding to the active Sticky Actions for Pass, Shot, Dribble, Sprint
        self.short_pass_index = 1
        self.long_pass_index = 2
        self.shot_index = 3
        self.dribble_index = 9
        self.sprint_index = 8

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Enhance the reward based on the presence of offensive actions.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, rew_value in enumerate(reward):
            o = observation[rew_index]
            actions = o['sticky_actions']
            
            # Reward for Short and Long Pass
            if actions[self.short_pass_index] or actions[self.long_pass_index]:
                components['pass_reward'][rew_index] = self.pass_reward
                reward[rew_index] += self.pass_reward
            
            # Reward for Shot
            if actions[self.shot_index]:
                components['shot_reward'][rew_index] = self.shot_reward
                reward[rew_index] += self.shot_reward
            
            # Reward for Dribble
            if actions[self.dribble_index]:
                components['dribble_reward'][rew_index] = self.dribble_reward
                reward[rew_index] += self.dribble_reward
            
            # Reward for Sprint
            if actions[self.sprint_index]:
                components['sprint_reward'][rew_index] = self.sprint_reward
                reward[rew_index] += self.sprint_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
