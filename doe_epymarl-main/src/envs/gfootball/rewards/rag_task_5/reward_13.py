import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on defensive skills training."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_zones = {}
        self.setup_transition_zones()
        self.transition_reward = 0.1
        
    def setup_transition_zones(self):
        """ Setup zones on the field that reward defensive position and quick counter-attacks """
        # Zones are defined in terms of x, y coordinates for simplification purposes
        self.transition_zones = {
            1: {"low": 0.0, "high": 0.2, "rewarded": False},  # Own goal area
            2: {"low": 0.2, "high": 0.4, "rewarded": False},  # Defense area
            3: {"low": 0.4, "high": 0.6, "rewarded": False},  # Midfield area
            4: {"low": 0.6, "high": 0.8, "rewarded": False},  # Offensive transition
            5: {"low": 0.8, "high": 1.0, "rewarded": False}   # Opponent's goal area
        }
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        for zone in self.transition_zones.values():
            zone["rewarded"] = False
        return self.env.reset()
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "transition_reward": [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            player_x = abs(player_pos[0])  # use absolute x-position to generalize to both team sides
            
            # Determine which zone the player is in and if it has not been already rewarded
            for zone_id, zone in self.transition_zones.items():
                if zone["low"] <= player_x < zone["high"] and not zone["rewarded"]:
                    components["transition_reward"][rew_index] = self.transition_reward
                    reward[rew_index] += components["transition_reward"][rew_index]
                    zone["rewarded"] = True
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Append component values into the info dictionary
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        obs = self.env.unwrapped.observation()
        # Update sticky actions counter
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = max(self.sticky_actions_counter[i], action)
        
        return observation, reward, done, info
