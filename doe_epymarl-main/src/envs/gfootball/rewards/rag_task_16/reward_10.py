import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward function to focus on high precision passes.
    It enhances the original reward by considering factors like high pass execution, trajectory control, and situational usage.
    """
    def __init__(self, env):
        super().__init__(env)
        self.high_pass_coefficient = 0.1  # Coefficient for high pass reward
        self.trajectory_control_coefficient = 0.05  # Coefficient for trajectory control
        self.situational_application_coefficient = 0.05  # Coefficient for situational application
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
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
        """
        Compute modified reward taking into account the trajectory and precision of high passes.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "trajectory_control_reward": [0.0] * len(reward),
                      "situational_application_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check for high passes: `ball_direction[2]` represents the z-component movement of the ball.
            # Consider high balls as those where the z-component is significantly positive.
            if o['ball_direction'][2] > 0.1:  # Assuming threshold for what constitutes a 'high' pass
                components["high_pass_reward"][rew_index] = self.high_pass_coefficient
                reward[rew_index] += components["high_pass_reward"][rew_index]
            
            # Reward trajectory control - we gauge this by the straightness of the trajectory to a teammate.
            # This is a simplified proxy measurement.
            if np.linalg.norm(o['ball_direction'][:2]) > 0.5:  # Rough estimate for control
                components["trajectory_control_reward"][rew_index] = self.trajectory_control_coefficient
                reward[rew_index] += components["trajectory_control_reward"][rew_index]
            
            # Situational application - evaluating the usefulness of a pass in specific match contexts.
            # For instance, deploying high passes to bypass defensive lines.
            if o['ball_owned_player'] != -1 and o['game_mode'] == 0:  # Normal play
                if np.abs(o['ball'][0]) > 0.5:  # Assuming the ball is in attacking half
                    components["situational_application_reward"][rew_index] = self.situational_application_coefficient
                    reward[rew_index] += components["situational_application_reward"][rew_index]
        
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
