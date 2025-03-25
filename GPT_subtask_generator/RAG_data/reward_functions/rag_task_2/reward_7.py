import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on rewarding defensive strategies and coordination."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Number of zones for defensive coordination
        self._num_zones = 5  
        self._zone_reward = 0.2
        # Tracking ball position entry into zones
        self.ball_zone_collect = np.zeros((self._num_zones,))
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_zone_collect.fill(0)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.ball_zone_collect.tolist()
        return self.env.get_state(to_pickle)
        
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_zone_collect = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle
        
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for i, o in enumerate(observation):
            # Calculate defensive zone based on ball position relative to y-axis and its ownership
            if o['ball_owned_team'] == 0:  # Defensive actions when ball is controlled by the opponent
                zone = int((o['ball'][1] + 0.42) / (0.84 / self._num_zones))
                if zone >= 0 and zone < self._num_zones and not self.ball_zone_collect[zone]:
                    self.ball_zone_collect[zone] = 1
                    reward[i] += self._zone_reward
                    components["defensive_reward"][i] = self._zone_reward
                    
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
