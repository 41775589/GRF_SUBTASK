import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for advanced dribbling and evasion techniques.
    Rewards the control of sprinting while maintaining ball possession and moving towards the opponent's goal.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.dribble_coefficient = 0.05
        self.sprint_coefficient = 0.03
        self.goal_approach_coefficient = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Restore any needed state
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward),
                      "goal_approach_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
    
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_index = o['active']
            
            # Calculate sprint rewards if sprint action is active
            if o['sticky_actions'][8] == 1:
                components["sprint_reward"][rew_index] = self.sprint_coefficient
            
            # Calculate dribble control rewards when dribbling with ball possession
            if o['sticky_actions'][9] == 1 and o['ball_owned_team'] == o['designated'] and o['ball_owned_player'] == active_player_index:
                components["dribble_reward"][rew_index] = self.dribble_coefficient
            
            # Reward progress towards opponent's goal (for attacking tactical training)
            if o['ball_owned_team'] == o['designated'] and o['ball_owned_player'] == active_player_index:
                ball_x = o['ball'][0]
                opponent_goal_x = 1 if o['designated'] == 0 else -1
                progress = abs(opponent_goal_x - ball_x)  # Distance to goal from current ball position
                components["goal_approach_reward"][rew_index] = (1 - progress) * self.goal_approach_coefficient
            
            # Combine all components
            total_additional_reward = (components["sprint_reward"][rew_index] +
                                       components["dribble_reward"][rew_index] +
                                       components["goal_approach_reward"][rew_index])
            reward[rew_index] += total_additional_reward
        
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
