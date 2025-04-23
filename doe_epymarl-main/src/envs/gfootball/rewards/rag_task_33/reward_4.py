import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for shooting from distance."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.distance_threshold = 0.6  # greater than this is considered a long-range shot
        self.long_range_reward = 0.5

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_range_shot_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            ball_pos = o['ball'][0]  # X axis position of the ball
            ball_owned_team = o['ball_owned_team']

            # Give additional reward for long-range shots
            if ball_owned_team == 0 and ball_pos > self.distance_threshold:
                components["long_range_shot_reward"][rew_index] = self.long_range_reward
                reward[rew_index] += components["long_range_shot_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                if 'sticky_actions' in agent_obs:
                    for i, action in enumerate(agent_obs['sticky_actions']):
                        self.sticky_actions_counter[i] += action
                        info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
