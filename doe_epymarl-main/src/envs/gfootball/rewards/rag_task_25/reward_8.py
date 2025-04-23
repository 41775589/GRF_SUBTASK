import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for dribbling techniques and sprint use."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_increment_reward = 0.05
        self.sprint_increment_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        obs = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}
        
        if obs is None:
            return reward, components
        
        assert len(reward) == len(obs)
        
        for i in range(len(reward)):
            o = obs[i]

            # Check if the player is dribbling.
            dribbling_active = o['sticky_actions'][9]  # index 9 corresponds to dribbling
            if dribbling_active:
                components["dribble_reward"][i] = self.dribble_increment_reward
                reward[i] += components["dribble_reward"][i]
                self.sticky_actions_counter[9] += 1

            # Check if the player is sprinting.
            sprinting_active = o['sticky_actions'][8]  # index 8 corresponds to sprinting
            if sprinting_active:
                components["sprint_reward"][i] = self.sprint_increment_reward
                reward[i] += components["sprint_reward"][i]
                self.sticky_actions_counter[8] += 1

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
