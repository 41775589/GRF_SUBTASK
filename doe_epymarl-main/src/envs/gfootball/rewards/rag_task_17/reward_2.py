import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward focusing on wide midfielder's ability to perform high passes and positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_checkpoints_collected = {}
        self.high_pass_reward = 0.2
        self.positioning_reward = 0.1
        self.num_position_checkpoints = 5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_checkpoints_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_position_checkpoints'] = self.position_checkpoints_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.position_checkpoints_collected = from_pickle['CheckpointRewardWrapper_position_checkpoints']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            active_player_position = obs['right_team'][obs['active']]
            # Reward high pass action (action 8 in sticky actions represents high_pass)
            if obs['sticky_actions'][8] == 1:
                components["high_pass_reward"][rew_index] = self.high_pass_reward
                reward[rew_index] += components["high_pass_reward"][rew_index]
                
            # Encourage positioning by dividing the x-axis into equal parts
            position_index = int(np.clip((active_player_position[0] * self.num_position_checkpoints + 1), 0, self.num_position_checkpoints-1))
            if position_index not in self.position_checkpoints_collected:
                self.position_checkpoints_collected[position_index] = True
                components["positioning_reward"][rew_index] = self.positioning_reward * (self.num_position_checkpoints - len(self.position_checkpoints_collected))
                reward[rew_index] += components["positioning_reward"][rew_index]
        
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
