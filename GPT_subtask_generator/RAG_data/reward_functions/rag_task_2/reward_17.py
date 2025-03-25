import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards for maintaining defensive positions while controlling the ball."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_checkpoints = [-0.8, -0.6, -0.4, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8]
        self.checkpoint_reward = 0.05
        self.active_player_last_pos = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.active_player_last_pos = {}
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.active_player_last_pos
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.active_player_last_pos = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning_reward": [0.0] * len(reward)}

        for i in range(len(reward)):
            o = observation[i]
            curr_position = o['right_team'][o['active']]
            self.active_player_last_pos.setdefault(i, curr_position[0])

            # Calculate reward for maintaining positions strategically
            if abs(self.active_player_last_pos[i] - curr_position[0]) in self.position_checkpoints:
                components["defensive_positioning_reward"][i] = self.checkpoint_reward
            
            # Adjust the main reward with the new defensive positioning reward
            reward[i] += components["defensive_positioning_reward"][i]
            self.active_player_last_pos[i] = curr_position[0]  

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
