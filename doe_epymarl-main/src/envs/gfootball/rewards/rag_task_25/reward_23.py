import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides rewards specialized for dribbling and sprinting."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribbling_proximity_reward = 0.05
        self.sprinting_bonus_reward = 0.03
        self.closeness_threshold = 0.2

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
                      "dribbling_proximity": [0.0] * len(reward),
                      "sprinting_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]
            if 'ball_owned_team' not in player_obs or player_obs['ball_owned_team'] != player_obs['active']:
                continue
            
            ball_position = player_obs['ball']
            player_position = player_obs['left_team'][player_obs['active']] if player_obs['active'] < len(player_obs['left_team']) else player_obs['right_team'][player_obs['active'] - len(player_obs['left_team'])]
            
            # Reward players for maintaining close ball proximity during dribbling
            if 'sticky_actions' in player_obs and player_obs['sticky_actions'][9] == 1: # 9 corresponds to dribble action
                distance_to_ball = np.sqrt((player_position[0] - ball_position[0])**2 + (player_position[1] - ball_position[1])**2)
                if distance_to_ball < self.closeness_threshold:
                    components["dribbling_proximity"][rew_index] = self.dribbling_proximity_reward
                    reward[rew_index] += components["dribbling_proximity"][rew_index]
            
            # Add a small reward for using sprint action effectively
            if 'sticky_actions' in player_obs and player_obs['sticky_actions'][8] == 1: # 8 corresponds to sprint action
                components["sprinting_bonus"][rew_index] = self.sprinting_bonus_reward
                reward[rew_index] += components["sprinting_bonus"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
