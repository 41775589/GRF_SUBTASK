import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """Wrapper to add a complex reward based on lateral field play and high pass skills."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.lateral_multiplier = 0.1
        self.high_pass_multiplier = 0.2
        self.possession_bonus = 0.05
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # The function must restore any state from `from_pickle`
        return from_pickle
    
    def reward(self, reward):
        obs = self.env.unwrapped.observation()
        if obs is None:
            return reward, {}

        # Modify rewards based on managing lateral plays and high pass utilizations
        components = {"base_score_reward": reward.copy(),
                      "lateral_play_bonus": [0.0] * len(reward),
                      "high_pass_bonus": [0.0] * len(reward),
                      "possession_bonus": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = obs[rew_index]
            # Reward for lateral player movement
            if 'left_team_roles' in o and 'right_team_roles' in o:
                left_wings = np.array([o['left_team'][i] for i in range(len(o['left_team_roles'])) if o['left_team_roles'][i] in [6,7]])  # LM, RM roles
                right_wings = np.array([o['right_team'][i] for i in range(len(o['right_team_roles'])) if o['right_team_roles'][i] in [6,7]])  # LM, RM roles
                
                if len(left_wings) > 0:
                    mean_y_position = np.mean(left_wings[:, 1])
                    components['lateral_play_bonus'][rew_index] = self.lateral_multiplier * np.abs(mean_y_position)
                
                if len(right_wings) > 0:
                    mean_y_position = np.mean(right_wings[:, 1])
                    components['lateral_play_bonus'][rew_index] += self.lateral_multiplier * np.abs(mean_y_position)

            # Reward for high passes
            if o['sticky_actions'][9] == 1:  # Check if 'high pass' action is used
                components['high_pass_bonus'][rew_index] = self.high_pass_multiplier

            # Bonus for ball possession in lateral areas
            if o['ball_owned_team'] == 0 or o['ball_owned_team'] == 1:  # If the ball is owned
                ball_y_pos = o['ball'][1]
                if abs(ball_y_pos) > 0.3:  # Ball is towards the sides of the pitch
                    components['possession_bonus'][rew_index] = self.possession_bonus
            
            # Update final adjusted reward
            reward[rew_index] += sum(components[k][rew_index] for k in components)
        
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
