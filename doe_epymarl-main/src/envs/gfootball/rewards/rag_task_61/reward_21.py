import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for maintaining possession during transition phases of the game."""
    
    def __init__(self, env):
        super().__init__(env)
        self.transition_reward = 0.05
        self.opponent_in_proximity_threshold = 0.1
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
        observation = self.env.unwrapped.observation()

        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            our_team = o['left_team'] if o['ball_owned_team'] == 0 else o['right_team']
            opponent_team = o['right_team'] if o['ball_owned_team'] == 0 else o['left_team']
            
            # Check if player is active and if ball is with the owned team
            if o['ball_owned_team'] in [0, 1] and o['active'] == o['ball_owned_player']:
                opponent_distance = np.min(np.linalg.norm(opponent_team - o['ball'], axis=1))
                
                if opponent_distance <= self.opponent_in_proximity_threshold:
                    components["transition_reward"][rew_index] = self.transition_reward
                    reward[rew_index] += components["transition_reward"][rew_index]
        
        return reward.copy(), components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions counter for diagnosis
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
