import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a tactical defensive reward focused on penalty area control and coordination."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.num_defensive_zones = 5  # Number of defensive zones defined near the penalty area
        self.defensive_rewards = np.linspace(0.1, 0.5, self.num_defensive_zones)
        self.zone_boundaries = np.linspace(-0.42, 0.42, self.num_defensive_zones + 1)
        self.visited_zones = [set(), set()]
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.visited_zones = [set(), set()]
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.visited_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.visited_zones = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        # Analyze per agent observation.
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            team = 'left' if o['ball_owned_team'] == 0 else 'right'
            player_pos = o[team + '_team'][o['active']]
            
            # Check if agent is near their penalty area and the other team controls the ball
            if o['ball_owned_team'] == 1 - rew_index:
                for zone_idx, boundary in enumerate(self.zone_boundaries[:-1]):
                    next_boundary = self.zone_boundaries[zone_idx + 1]
                    # Check if player is in the current defensive zone
                    if boundary <= player_pos[1] <= next_boundary:
                        # Reward is assigned based on the zone and whether it's the first visit
                        if zone_idx not in self.visited_zones[rew_index]:
                            self.visited_zones[rew_index].add(zone_idx)
                            reward_delta = self.defensive_rewards[zone_idx]
                            components["defensive_reward"][rew_index] += reward_delta
                            reward[rew_index] += reward_delta

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
