import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a rich reward for mastering defensive tactics
       during diverse gameplay scenarios:
       - Encourages successful tackles
       - Penalizes fouls
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_reward = 0.3
        self.foul_penalty = -0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "foul_penalty": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            tackle_action = any(o['sticky_actions'][3:5])  # indices for tackle actions
            foul_committed = o['game_mode'] in {3, 6}  # Assumed game_modes for fouls and penalties

            if tackle_action:
                components["tackle_reward"][rew_index] = self.tackle_success_reward
                reward[rew_index] += self.tackle_success_reward

            if foul_committed:
                components["foul_penalty"][rew_index] = self.foul_penalty
                reward[rew_index] += self.foul_penalty
        
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
