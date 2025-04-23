import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards strategic positioning and coordination in midfield 
    play during transitions between offense and defense.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define midfield zones based on normalized x-positions
        self.midfield_zones = [(0.2, 0.4), (0.4, 0.6), (0.6, 0.8)]
        self.zone_rewards = [0.1, 0.2, 0.1]
        self.visits = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.visits = {zone: False for zone in self.midfield_zones}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "positioning_reward": [0.0]
        }
        
        if observation is None:
            return reward, components
        
        position_x = observation[0]['left_team'][observation[0]['active']][0]  # Active player's x-coordinate
        
        for idx, zone in enumerate(self.midfield_zones):
            if zone[0] <= position_x < zone[1] and not self.visits[zone]:
                reward[0] += self.zone_rewards[idx]
                components["positioning_reward"][0] += self.zone_rewards[idx]
                self.visits[zone] = True

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
                self.sticky_actions_counter[i] = action
        
        return observation, reward, done, info
