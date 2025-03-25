import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on wide midfield responsibilities, mastering high passes and positioning."""
    
    def __init__(self, env):
        super().__init__(env)
        self._high_pass_reward = 0.2  # Reward multiplier for successful high passes
        self._positioning_reward = 0.1  # Reward for optimal positioning
        self._cross_halfway_reward = 0.1  # Reward for crossing the halfway line horizontally
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "high_pass_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward),
                      "cross_halfway_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_pos = o['right_team'][o['active']] if o['ball_owned_team'] == 1 else o['left_team'][o['active']]

            # Check for high pass action outcome
            if o['sticky_actions'][9] == 1:  # Assuming index 9 refers to high pass action
                components['high_pass_reward'][rew_index] = self._high_pass_reward
                reward[rew_index] += components['high_pass_reward'][rew_index]

            # Check for positioning reward
            if abs(active_player_pos[1]) < 0.1:  # Good lateral positioning near center
                components['positioning_reward'][rew_index] = self._positioning_reward
                reward[rew_index] += components['positioning_reward'][rew_index]

            # Check for crossing the halfway line reward
            if abs(active_player_pos[0]) > 0.5:  # Crossed the horizontal halfway line
                components['cross_halfway_reward'][rew_index] = self._cross_halfway_reward
                reward[rew_index] += components['cross_halfway_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
                
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
