import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward focusing on defensive coordination near the penalty area."""

    def __init__(self, env):
        super().__init__(env)
        self.defensive_zones_covered = set()
        # Define regions near the penalty area as critical defensive zones
        self.critical_defensive_zones = [
            (-1, -0.20), (-0.9, -0.20), (-0.8, -0.20),
            (-1, 0.20), (-0.9, 0.20), (-0.8, 0.20)
        ]
        self.zone_coverage_reward = 0.05  # Reward increment for covering a new zone
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.defensive_zones_covered = set()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['DefensiveZonesCovered'] = self.defensive_zones_covered
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_zones_covered = from_pickle['DefensiveZonesCovered']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            player_pos = observation[rew_index]['left_team'] \
                if observation[rew_index]['active'] in observation[rew_index]['left_team'] \
                else observation[rew_index]['right_team']
            ball_pos = observation[rew_index]['ball'][:2]  # Exclude the z-coordinate

            for zone in self.critical_defensive_zones:
                if np.linalg.norm(np.array(player_pos) - np.array(zone)) < 0.1:
                    if zone not in self.defensive_zones_covered:
                        self.defensive_zones_covered.add(zone)
                        reward[rew_index] += self.zone_coverage_reward

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
