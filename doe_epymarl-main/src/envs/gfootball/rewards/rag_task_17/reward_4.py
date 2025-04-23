import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to train agents on mastering wide midfield responsibilities, 
    focusing on high pass execution and effective positioning to facilitate lateral transitions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.wide_midfield_coefficient = 0.2
        self.high_pass_coefficient = 0.1
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10))
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "wide_midfield_reward": [0.0] * len(reward),
                      "high_pass_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for i in range(len(reward)):
            current_obs = observation[i]
            
            midfield_positional_reward = 0.0
            sideline_position_x_thresholds = (-0.6, 0.6)  # Assuming this approximates "wide" midfield
            sideline_position_y_thresholds = (-0.2, 0.2)  # Players should also not be too close/far from center

            player_position = current_obs['left_team'][current_obs['active']] if current_obs['active'] < len(current_obs['left_team']) else current_obs['right_team'][current_obs['active'] - len(current_obs['left_team'])]

            # Check if the player is in a wide and effective midfield position
            if (sideline_position_x_thresholds[0] <= player_position[0] <= sideline_position_x_thresholds[1] and
                sideline_position_y_thresholds[0] <= player_position[1] <= sideline_position_y_thresholds[1]):
                midfield_positional_reward = self.wide_midfield_coefficient

            components["wide_midfield_reward"][i] = midfield_positional_reward

            # Reward for executing a high pass correctly
            if current_obs['sticky_actions'][2]:  # Assuming index 2 corresponds to high pass action
                direction = current_obs['right_team_direction' if current_obs['ball_owned_team'] == 1 else 'left_team_direction'][current_obs['active']]
                if np.linalg.norm(direction) > 0.5:  # Assume using directional intensity to estimate "high" power in pass
                    components["high_pass_reward"][i] = self.high_pass_coefficient
            
            reward[i] += components["wide_midfield_reward"][i] + components["high_pass_reward"][i]
        
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
            for idx, action in enumerate(agent_obs['sticky_actions']):
                # Updating sticker action counters per agent
                self.sticky_actions_counter[idx] += action
        return observation, reward, done, info
