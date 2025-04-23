import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for training wingers' crossing and sprinting abilities."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._num_crossings = 10
        self._crossing_reward = 0.1
        self._collected_crossings = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_crossings = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_crossings
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_crossings = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "crossing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if ('ball_owned_team' not in o or
                    o['ball_owned_team'] != o['active'] or
                    'sticky_actions' not in o or
                    not o['sticky_actions'][9]):  # Check if dribble action is active
                continue

            # Calculate the distance from player to crossing zone (wings)
            player_pos = o['left_team'][o['active'] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]]
            crossing_zone_y = 0.42 if player_pos[1] > 0 else -0.42  # Assume top corners are ideal crossing zones
            distance_to_crossing_zone = abs(crossing_zone_y - player_pos[1])

            # Reward for entering crossing regions
            if distance_to_crossing_zone < 0.1 and (self._collected_crossings.get(rew_index, 0) < self._num_crossings):
                components["crossing_reward"][rew_index] = self._crossing_reward
                reward[rew_index] += components["crossing_reward"][rew_index]
                self._collected_crossings[rew_index] = self._collected_crossings.get(rew_index, 0) + 1

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
