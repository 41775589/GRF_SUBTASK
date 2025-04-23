import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward to enhance defensive play and counterattack strategy."""

    def __init__(self, env):
        super().__init__(env)
        self._num_zones = 5  # Number of zones for strategic positioning
        self._zone_reward = 0.05  # Reward for reaching a new zone effectively
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_zones = {}

    def reset(self):
        self._collected_zones = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_zones = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_zone_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            # Zones are defined based on defensive and counterattack positions:
            player_x, player_y = o['right_team'][o['designated']]
            threshold = np.linspace(0.0, 0.5, self._num_zones + 1)[1:-1]  # exclude 0 and goalie-line

            # Calculate zone based on x-coordinate position, which reflects strategic depth
            current_zone = np.digitize(player_x, threshold)

            # Reward the player first reaching each strategic zone
            max_zone_reached = self._collected_zones.get(i, 0)
            if current_zone > max_zone_reached:
                components["defensive_zone_reward"][i] = self._zone_reward
                reward[i] += self._zone_reward
                self._collected_zones[i] = current_zone

        # Return modified reward and components detailing the rewards
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        if observation:
            self.sticky_actions_counter.fill(0)
            for agent_obs in observation:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
