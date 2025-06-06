import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive actions in football."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._successful_tackles = {}
        self._tackle_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._successful_tackles = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._successful_tackles
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._successful_tackles = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            controlled_player_pos = np.array(o['left_team'][o['active']])
            closest_opponent_idx = np.argmin(np.linalg.norm(
                o['right_team'] - controlled_player_pos, axis=-1))

            closest_distance = np.linalg.norm(
                o['right_team'][closest_opponent_idx] - controlled_player_pos)

            if closest_distance < 0.015 and o['ball_owned_team'] == 1:
                if (rew_index not in self._successful_tackles or
                        self._successful_tackles[rew_index] < 3):
                    components["tackle_reward"][rew_index] = self._tackle_reward
                    reward[rew_index] += components["tackle_reward"][rew_index]
                    self._successful_tackles[rew_index] = (
                        self._successful_tackles.get(rew_index, 0) + 1)

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
