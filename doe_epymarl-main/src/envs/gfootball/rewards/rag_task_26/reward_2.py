import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for midfield dynamics focusing on central and wide contributions."""

    def __init__(self, env):
        super().__init__(env)
        self._central_midfield_control = {}
        self._wide_midfield_control = {}
        self._num_zones = 10
        self._zone_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._central_midfield_control = {}
        self._wide_midfield_control = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['central_midfield_control'] = self._central_midfield_control
        to_pickle['wide_midfield_control'] = self._wide_midfield_control
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._central_midfield_control = from_pickle['central_midfield_control']
        self._wide_midfield_control = from_pickle['wide_midfield_control']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "central_midfield_reward": np.zeros(len(reward)),
            "wide_midfield_reward": np.zeros(len(reward))
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Process each agent's observation
        for agent_index, obs in enumerate(observation):
            # Get midfield positions and check control
            central_zone_range = (-0.1, 0.1)  # Define the central zone on the field
            wide_zone_range = ((-1.0, -0.3), (0.3, 1.0))  # Define the wide zones

            player_pos = obs['right_team' if obs['active'] in obs['right_team_roles'] else 'left_team'][obs['active']]
            x, y = player_pos

            if central_zone_range[0] <= x <= central_zone_range[1]:
                if agent_index not in self._central_midfield_control:
                    self._central_midfield_control[agent_index] = self._num_zones
                    components["central_midfield_reward"][agent_index] = self._zone_reward
                    reward[agent_index] += components["central_midfield_reward"][agent_index]

            if wide_zone_range[0][0] <= x <= wide_zone_range[0][1] or wide_zone_range[1][0] <= x <= wide_zone_range[1][1]:
                if agent_index not in self._wide_midfield_control:
                    self._wide_midfield_control[agent_index] = self._num_zones
                    components["wide_midfield_reward"][agent_index] = self._zone_reward
                    reward[agent_index] += components["wide_midfield_reward"][agent_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
