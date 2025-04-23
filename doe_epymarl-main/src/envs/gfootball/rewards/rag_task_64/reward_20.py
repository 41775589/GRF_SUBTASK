import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on successful high passes and cross-field plays to promote dynamic attacking."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_passes = 0
        self.cross_plays = 0
        self.pass_reward = 0.1
        self.cross_play_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_passes = 0
        self.cross_plays = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'high_passes': self.high_passes, 'cross_plays': self.cross_plays}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.high_passes = from_pickle['CheckpointRewardWrapper']['high_passes']
        self.cross_plays = from_pickle['CheckpointRewardWrapper']['cross_plays']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "cross_play_reward": [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] == 5:  # Assuming mode 5 is high pass
                components["pass_reward"][rew_index] = self.pass_reward
                self.high_passes += 1

            if abs(o['ball'][0]) > 0.5 and o['ball_owned_team'] == o['active']:  # Cross-field attempt
                components["cross_play_reward"][rew_index] = self.cross_play_reward
                self.cross_plays += 1

            reward[rew_index] += components["pass_reward"][rew_index] + components["cross_play_reward"][rew_index]

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
