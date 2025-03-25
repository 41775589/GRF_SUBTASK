import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive coordination near the penalty area."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_zones_collect = {}
        self.num_zones = 5
        self.zone_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_zones_collect = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.defensive_zones_collect
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_zones_collect = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_zone_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check for defensive actions near the penalty area
            if o['game_mode'] in [2, 4]:  # Focusing game modes for Corner and GoalKick
                player_pos = o['left_team'] if o['active'] in o['left_team_active'] else o['right_team']
                active_player = o['active']
                defensive_zone_index = player_pos[active_player][0] * self.num_zones

                if self.defensive_zones_collect.get(active_player, -1) < defensive_zone_index:
                    components['defensive_zone_reward'][rew_index] = self.zone_reward
                    reward[rew_index] += components['defensive_zone_reward'][rew_index]
                    self.defensive_zones_collect[active_player] = defensive_zone_index

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
