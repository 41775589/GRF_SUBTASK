import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward based on defense performance at critical regions near the penalty area."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward increments for achieving a good defensive position
        self.defensive_position_reward = 0.05
        # High pressure scenario bonus
        self.high_pressure_defense_reward = 0.1
        # Defensive synergy multiplier
        self.defensive_synergy_multiplier = 2
        # Keep track of key defensive areas covered (e.g. near penalty area)
        self.covered_areas = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.covered_areas = []
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_position_reward": [0.0] * len(reward),
            "high_pressure_defense_reward": [0.0] * len(reward)
        }

        for rew_index, original_reward in enumerate(reward):
            o = observation[rew_index]
            key_defensive_areas = self.identify_defensive_zones(o)
            player_position = [o[player] for player in ('left_team', 'right_team') if o.get(player)]

            for zone in key_defensive_areas:
                if player_position == zone and zone not in self.covered_areas:
                    components["defensive_position_reward"][rew_index] = self.defensive_position_reward
                    reward[rew_index] += components["defensive_position_reward"][rew_index]
                    self.covered_areas.append(zone)

            if self.is_high_pressure(o):
                components["high_pressure_defense_reward"][rew_index] = self.high_pressure_defense_reward
                reward[rew_index] += components["high_pressure_defense_reward"][rew_index] 

            # Apply synergy multiplier if multiple agents meet criteria
            if len(self.covered_areas) > 1:
                reward[rew_index] *= self.defensive_synergy_multiplier

        return reward, components

    def is_high_pressure(self, obs):
        # Simulating a condition where high defensive pressure is identifiable
        return obs.get('ball_owned_team', -1) == 1 and np.abs(obs['ball'][1]) < 0.05

    def identify_defensive_zones(self, obs):
        # Placeholder zones representing sensitive defensive areas near the penalty area
        return [(-0.8, 0.2), (-0.9, 0)]

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.covered_areas
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.covered_areas = from_pickle['CheckpointRewardWrapper']
        return from_pickle

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
