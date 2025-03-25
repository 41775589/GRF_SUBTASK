import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on offensive actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define offensive actions: Short Pass, Long Pass, Shot, Dribble, Sprint with indices
        self.offensive_actions = {
            'short_pass': 0,
            'long_pass': 1,
            'shot': 2,
            'dribble': 8,
            'sprint': 9
        }
        self.action_rewards = {
            'short_pass': 0.1,
            'long_pass': 0.2,
            'shot': 0.3,
            'dribble': 0.05,
            'sprint': 0.05
        }
    
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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_action_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Process sticky actions for offensive rewards.
            for action_name, action_idx in self.offensive_actions.items():
                if o['sticky_actions'][action_idx]:
                    components["offensive_action_reward"][rew_index] += self.action_rewards[action_name]

            reward[rew_index] += components["offensive_action_reward"][rew_index]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
