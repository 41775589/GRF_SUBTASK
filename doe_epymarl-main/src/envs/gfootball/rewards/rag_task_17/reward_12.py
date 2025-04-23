import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes agents to utilize wide midfield areas and execute high passes effectively."""
    
    def __init__(self, env):
        super().__init__(env)
        self._num_zonal_checks = 6  # Number of zones across the width
        self._zone_rewards_collected = {}
        self.horizontal_zones = np.linspace(-0.42, 0.42, self._num_zonal_checks + 1)
        self.high_pass_action = 9  # Assuming that action 9 corresponds to the high pass
        self.zoom_coefficient = 0.1  # Reward scaling for successful high pass in the zone

        # Sticky actions tracking per zone
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self._zone_rewards_collected = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._zone_rewards_collected
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._zone_rewards_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "zonal_positioning_reward": [0.0] * len(reward),
                      "high_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            active_player_position = o['left_team'][o['active']]
            active_player_y = active_player_position[1]  # Y coordinate of the active player

            # Check which zone the active player is in
            zonal_index = np.digitize([active_player_y], self.horizontal_zones) - 1
            
            if zonal_index < 0 or zonal_index >= self._num_zonal_checks:
                continue

            # Check and reward if the player is performing a high pass in the zone
            if o['sticky_actions'][self.high_pass_action]:
                if zonal_index not in self._zone_rewards_collected:
                    components["high_pass_reward"][rew_index] = self.zoom_coefficient
                    reward[rew_index] += components["high_pass_reward"][rew_index]
                    self._zone_rewards_collected[zonal_index] = True
                
            components["zonal_positioning_reward"][rew_index] = zonal_index / self._num_zonal_checks
            reward[rew_index] += components["zonal_positioning_reward"][rew_index]
        
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
