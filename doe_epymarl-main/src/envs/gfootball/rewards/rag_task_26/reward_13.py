import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds midfield dynamic reward based on player positions and roles."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define various strategic zones on the pitch to encourage position play
        self.midfield_zones = {
            'central_zone': (0.2, -0.2, 0.3, -0.3),  # Encourage CM play
            'wide_zones': ((0.4, 0.6), (-0.4, -0.6)), # Encourage LM, RM play
        }
        self.midfield_reward_coefficient = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team' if o['active'] in o['left_team_roles'] else 'right_team'][o['active']]
            team_x, team_y = player_pos[0], player_pos[1]
            
            # Central midfield roles control
            if o['active'] in (4, 5, 6) and self.midfield_zones['central_zone'][0] <= team_x <= self.midfield_zones['central_zone'][1] and self.midfield_zones['central_zone'][2] <= team_y <= self.midfield_zones['central_zone'][3]:
                components["midfield_control_reward"][rew_index] = self.midfield_reward_coefficient
            
            # Wide midfield roles control
            if o['active'] in (6, 7) and any(self.midfield_zones['wide_zones'][0][0] <= team_x <= self.midfield_zones['wide_zones'][0][1] or self.midfield_zones['wide_zones'][1][0] <= team_x <= self.midfield_zones['wide_zones'][1][1]) and abs(team_y) >= 0.35:
                components["midfield_control_reward"][rew_index] += self.midfield_reward_coefficient

            # Total reward computation
            reward[rew_index] += components["midfield_control_reward"][rew_index]
        
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
