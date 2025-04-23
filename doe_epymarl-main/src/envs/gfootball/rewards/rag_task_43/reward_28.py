import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward focusing on defensive maneuvers and quick counterattacks.
    This reward function is designed to encourage agents to improve their defensive positioning
    and responsiveness, while also rewarding efficient transitions into counterattacks.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initiate defensive and counterattack rewards settings
        self.defensive_positions = np.array([[-1, 0], [-0.75, 0], [-0.5, 0], [-0.25, 0]])
        self.defensive_reward_coefficient = 0.1
        self.counterattack_reward_coefficient = 0.2
        
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
        components = {"base_score_reward": reward.copy(), "defensive_reward": [0.0] * len(reward), "counterattack_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for index, o in enumerate(observation):
            # Calculate defensive reward based on player's position relative to defensive strategy positions
            player_position = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            if np.any(np.linalg.norm(self.defensive_positions - player_position, axis=1) < 0.1):
                components["defensive_reward"][index] = self.defensive_reward_coefficient
            
            # Calculate counterattack reward based on quick transition into attacking phase
            ball_direction = o['ball_direction']
            own_goal_direction = -1 if o['ball_owned_team'] == 0 else 1
            if own_goal_direction * ball_direction[0] > 0:  # Ball moving towards opponent's goal quickly
                components["counterattack_reward"][index] = self.counterattack_reward_coefficient
            
            reward[index] += components["defensive_reward"][index] + components["counterattack_reward"][index]
        
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
