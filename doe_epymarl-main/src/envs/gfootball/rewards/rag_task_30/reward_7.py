import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for improved defensive strategies and transitions to counterattacks."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_positions_collected = set()
        self._defensive_reward = 0.01
        self._transition_bonus = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_positions_collected = set()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = list(self._defensive_positions_collected)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._defensive_positions_collected = set(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        # Base reward not modified, used only to keep structure
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_position_reward": [0.0] * len(reward),
            "transition_bonus_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        # Custom reward logic here
        assert len(reward) == len(observation)
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for achieving certain defensive positions without current ball possession.
            defensive_positions = [(0.1, 0), (-0.1, 0), (0., 0.1), (0., -0.1)]
            current_position = (o['left_team'][o['active']][0], o['left_team'][o['active']][1])

            for defensive_position in defensive_positions:
                if current_position == defensive_position and defensive_position not in self._defensive_positions_collected:
                    self._defensive_positions_collected.add(defensive_position)
                    reward[rew_index] += self._defensive_reward
                    components["defensive_position_reward"][rew_index] = self._defensive_reward

            # Additional reward if transitioning from defense to counterattack
            if o['ball_owned_team'] == 0 and current_position == (0.0, 0.0):
                reward[rew_index] += self._transition_bonus
                components["transition_bonus_reward"][rew_index] = self._transition_bonus

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
