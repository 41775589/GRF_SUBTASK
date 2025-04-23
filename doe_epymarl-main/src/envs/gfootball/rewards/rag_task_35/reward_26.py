import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for strategic positioning and movements."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize specific attributes for reward calculation
        self.init_reward_params()
        
    def init_reward_params(self):
        # Weight for strategic positioning (near the ball or relevant field positions)
        self.positioning_weight = 0.05
        # Weight for effective switching between attack and defense
        self.switching_weight = 0.1
        # Define key regions in the field for strategic positioning
        self.key_regions = [(-0.5, 0), (0, 0), (0.5, 0)]  # center left, center, center right
    
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
                      "positioning_reward": [0.0] * len(reward),
                      "switching_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            player_pos = o['right_team'] if o['active'] in o['right_team_active'] else o['left_team']
            ball_pos = o['ball'][:2]  # only use x, y
            
            # Evaluate strategic positioning
            closest_key_region = min(self.key_regions, key=lambda x: np.linalg.norm(np.array(x) - np.array(ball_pos)))
            dist_to_key_region = np.linalg.norm(np.array(player_pos[o['active']]) - np.array(closest_key_region))
            components["positioning_reward"][rew_index] = self.positioning_weight / (1 + dist_to_key_region)
            
            # Evaluate effectiveness of switching between tasks
            if rew_index > 0:  # Assuming the previous index corresponds to the previous time step
                previous_observation = observation[rew_index - 1]
                prev_ball_pos = previous_observation['ball'][:2]
                movement_direction_is_towards_ball = np.dot(ball_pos - prev_ball_pos, player_pos[o['active']] - ball_pos) > 0
                if movement_direction_is_towards_ball:
                    components["switching_reward"][rew_index] = self.switching_weight
            
            reward[rew_index] += components["positioning_reward"][rew_index] + components["switching_reward"][rew_index]
        
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
