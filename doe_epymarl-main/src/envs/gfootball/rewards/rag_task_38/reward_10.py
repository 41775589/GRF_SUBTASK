import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to encourage counterattacks from defense via long passes and quick transitions.
    This is specialized for tasks where initiating counterattacks is crucial after gaining possession in defensive zones.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_reward_coef = 0.5  # Coefficient for long pass reward
        self.quick_transition_reward_coef = 0.3  # Coefficient for quick transition reward
        self.previous_ball_position = None
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['previous_ball_position'] = self.previous_ball_position
        return super().get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = super().set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        self.previous_ball_position = from_pickle.get('previous_ball_position')
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward),
                      "quick_transition_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball'][:2]
            own_team_control = o['ball_owned_team'] == (0 if o['left_team_active'].sum() > 0 else 1)
            
            # Check for successful long pass
            if self.previous_ball_position is not None and own_team_control:
                distance = np.linalg.norm(np.array(ball_position) - np.array(self.previous_ball_position))
                # Assuming a long pass is identified by a change in ball position by more than 0.5 on field
                if 'ball_owned_player' in o and distance > 0.5:
                    components["long_pass_reward"][rew_index] = self.long_pass_reward_coef * distance
            
            # Check for quick transition:
            # Assuming if ball moves quickly towards opponent half from our defensive half
            if self.previous_ball_position is not None and own_team_control:
                motion_vector = np.array(ball_position) - np.array(self.previous_ball_position)
                if self.previous_ball_position[0] < 0 and ball_position[0] > self.previous_ball_position[0]:
                    components["quick_transition_reward"][rew_index] = self.quick_transition_reward_coef * abs(motion_vector[0]) 
            
            # Update rewards with components
            reward[rew_index] += components["long_pass_reward"][rew_index] + components["quick_transition_reward"][rew_index]
            self.previous_ball_position = ball_position.copy()
        
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
