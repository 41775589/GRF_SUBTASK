import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that promotes quick decision-making and efficient ball handling 
       for counter-attacks immediately after recovery of ball possession."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.counter_attacks_reward = 0.5
        self.possession_recovery_reward = 0.35
        self.previous_ball_owner = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.previous_ball_owner = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
        
    def get_state(self, to_pickle):
        to_pickle['previous_ball_owner'] = self.previous_ball_owner
        return self.env.get_state(to_pickle)
        
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_owner = from_pickle['previous_ball_owner']
        return from_pickle
        
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward.copy()}
        
        modified_reward = reward.copy()
        components = {"base_score_reward": reward.copy(), 'possession_recovery_reward': 0.0, 'counter_attacks_reward': 0.0}
        
        # Evaluate the important events
        ball_owner_team = observation['ball_owned_team']
        if ball_owner_team != self.previous_ball_owner:
            if ball_owner_team == 0:  # Assuming the agent team is team '0'
                components['possession_recovery_reward'] = self.possession_recovery_reward
                modified_reward += self.possession_recovery_reward
        
        # Bonus for moving quickly towards the opponent's goal with the ball
        if ball_owner_team == 0:  # Ball is with agent's team
            distance_to_goal = 1 - observation['ball'][0]  # 'ball' has the position x, y, z where x is along the field
            if distance_to_goal < 0.5:  # Quick advance towards the goal (half field crossed)
                components['counter_attacks_reward'] = self.counter_attacks_reward
                modified_reward += self.counter_attacks_reward
        
        self.previous_ball_owner = ball_owner_team
        return modified_reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = reward
        for key, value in components.items():
            info[f"component_{key}"] = value
        
        # Update info with sticky actions and other potentially useful infos
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i in range(len(agent_obs['sticky_actions'])):
                if agent_obs['sticky_actions'][i]:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = agent_obs['sticky_actions'][i]
        
        return observation, reward, done, info
