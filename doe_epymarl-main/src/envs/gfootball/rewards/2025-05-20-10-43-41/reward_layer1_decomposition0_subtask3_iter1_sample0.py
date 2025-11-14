import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for encouraging effective offensive counter-attacks utilizing dribbling and long passes."""

    def __init__(self, env):
        super().__init__(env)
        self.counter_long_passes = 0.0
        self.counter_dribbling = 0.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {"sticky_actions_counter": self.sticky_actions_counter.copy()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward long passes that switch play or advance towards opponent's third
            if 'ball_direction' in o and abs(o['ball_direction'][0]) > 0.2 and o['ball_owned_team'] == 1:
                components["long_pass_reward"][rew_index] = 0.3
                reward[rew_index] += components["long_pass_reward"][rew_index]
                self.counter_long_passes += 1
            
            # Reward dribbling in own half to evade opponent pressure
            if (o['sticky_actions'][9] == 1 and o['ball'][0] < -0.5 and o['ball_owned_team'] == 1):
                components["dribbling_reward"][rew_index] = 0.3
                reward[rew_index] += components["dribbling_reward"][rew_index]
                self.counter_dribbling += 1

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
