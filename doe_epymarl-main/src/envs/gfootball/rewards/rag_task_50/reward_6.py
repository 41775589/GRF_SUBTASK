import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for executing accurate long passes over specific distances."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.long_pass_threshold = 0.3  # Define a threshold for distance which considers the pass as a long pass
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['last_ball_position'] = self.last_ball_position
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['last_ball_position']
        return from_pickle

    def reward(self, reward):
        # Initialize reward components dictionary
        components = {
            "base_score_reward": reward.copy(),
            "long_pass_reward": [0.0] * len(reward)
        }
        
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if self.last_ball_position is not None:
                current_ball_position = np.array(o['ball'][:2])
                last_ball_position = np.array(self.last_ball_position)
                pass_distance = np.linalg.norm(current_ball_position - last_ball_position)
                
                # Check if long pass criteria are met
                if pass_distance > self.long_pass_threshold:
                    if o['ball_owned_team'] == 0:  # ball is owned by our team after pass
                        components["long_pass_reward"][rew_index] = pass_distance * 0.1  # reward proportional to distance
                        reward[rew_index] += components["long_pass_reward"][rew_index]
            
            # Store current ball position for next step comparison
            self.last_ball_position = np.array(o['ball'][:2])
            
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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
