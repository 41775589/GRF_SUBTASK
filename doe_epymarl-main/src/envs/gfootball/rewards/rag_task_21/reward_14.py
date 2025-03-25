import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive actions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_zones = {}
        self._defensive_reward = 0.1
        self._zone_thresholds = np.linspace(-1, 1, 10)  # 10 zones along the x-axis from -1 to 1

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_zones = {}
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.defensive_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_zones = from_pickle.get('CheckpointRewardWrapper', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Immediate reward for blocking an opponent
            if o['game_mode'] in [3, 4, 6]:  # FreeKick, Corner, Penalty where defensive actions are crucial
                if o['ball_owned_team'] == 1:  # Ball is with the opponent
                    zone = np.digitize([o['ball'][0]], self._zone_thresholds)[0]
                    if zone not in self.defensive_zones.get(rew_index, []):
                        components["defensive_reward"][rew_index] += self._defensive_reward
                        reward[rew_index] += components["defensive_reward"][rew_index]
                        if rew_index in self.defensive_zones:
                            self.defensive_zones[rew_index].append(zone)
                        else:
                            self.defensive_zones[rew_index] = [zone]

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
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
