import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive tactic reward focusing on coordination near the penalty area."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._high_pressure_zones = {}
        self._defensive_coordination_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._high_pressure_zones = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['HighPressureZones'] = self._high_pressure_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._high_pressure_zones = from_pickle['HighPressureZones']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_coordination_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Focus on defending near the penalty area
            if (('left_team_roles' in o and o['left_team_roles'][o['active']] == 1) or
                ('right_team_roles' in o and o['right_team_roles'][o['active']] == 1)):
                if o['ball'][0] > 0.7 or o['ball'][0] < -0.7:  # Close to either goal side
                    zone_key = f"{rew_index}-{int(o['ball'][0] * 10)}"
                    if self._high_pressure_zones.get(zone_key, False) == False:
                        self._high_pressure_zones[zone_key] = True
                        components["defensive_coordination_reward"][rew_index] = self._defensive_coordination_reward
                        reward[rew_index] += components["defensive_coordination_reward"][rew_index]

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
