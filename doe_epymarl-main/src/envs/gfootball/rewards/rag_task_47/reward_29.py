import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on sliding tackles during high-pressure defensive situations."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_reward = 0.5  # Reward given for successful tackles
        self._pressure_zones = {
            (0.0, -0.25): 0.3,  # closer to goalie zone, higher pressure
            (-0.25, -0.5): 0.2,
            (-0.5, -0.75): 0.1,
            (-0.75, -1.0): 0.05
        }  # Define zones close to the own goal where pressure is higher

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            if 'left_team' in o:
                player_y_position = o['left_team'][o['active']][1]
                for zone in self._pressure_zones:
                    if zone[0] < player_y_position <= zone[1]:
                        if o['sticky_actions'][6] == 1:  # action_bottom performs a slide tackle
                            components["tackle_reward"][rew_index] = self._tackle_reward + self._pressure_zones[zone]
                            reward[rew_index] += components["tackle_reward"][rew_index]
                            break

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
