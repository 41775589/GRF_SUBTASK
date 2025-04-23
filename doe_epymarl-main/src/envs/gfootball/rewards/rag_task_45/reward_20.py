import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on stopping and starting movements."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward for successfully stopping or making sudden direction changes
        self.stop_and_change_reward = 0.1

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
                      "stop_change_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            active_player = obs['active']
            
            # Monitor significant changes in player direction
            if obs['sticky_actions'][3] == 1 or obs['sticky_actions'][4] == 1:  # top_left or top_right for sudden movement
                if self.sticky_actions_counter[rew_index] == 0:  # previously not moving in that direction
                    components["stop_change_reward"][rew_index] = self.stop_and_change_reward
                    reward[rew_index] += components["stop_change_reward"][rew_index]
                    self.sticky_actions_counter[rew_index] = 1

            # Monitor stops - changes from moving to idle
            elif all(action == 0 for action in obs['sticky_actions'][0:4]):  # checking movement action states
                if self.sticky_actions_counter[rew_index] == 1:  # previously moving
                    components["stop_change_reward"][rew_index] = self.stop_and_change_reward
                    reward[rew_index] += components["stop_change_reward"][rew_index]
                    self.sticky_actions_counter[rew_index] = 0

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
