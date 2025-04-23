import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to encourage wide midfield play using high passes and wide positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._high_pass_reward = 0.2  # Reward for performing a high pass
        self._positional_reward = 0.1  # Reward for maintaining a position wide on the field

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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "positional_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Reward for executing high passes
            if o['sticky_actions'][9] == 1:  # Check if 'high_pass' action is performed
                components["high_pass_reward"][rew_index] = self._high_pass_reward
                reward[rew_index] += components["high_pass_reward"][rew_index]
            
            # Reward for maintaining position on the wide areas of the field
            x_position = o['right_team'][o['active']][0]  # x position of the active player
            if abs(x_position) > 0.7:  # Check if the player is wide on the field
                components["positional_reward"][rew_index] = self._positional_reward
                reward[rew_index] += components["positional_reward"][rew_index]
        
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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
