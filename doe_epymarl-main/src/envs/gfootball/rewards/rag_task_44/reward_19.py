import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a stop-dribble control reward under pressure."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pressure_threshold = 0.2  # Example threshold
        self.stop_dribble_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_dribble_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            is_under_pressure = self.is_under_pressure(o)
            has_stopped_dribbling = self.is_stopped_dribbling(o['active'], o['sticky_actions'])

            if is_under_pressure and has_stopped_dribbling:
                reward[rew_index] += self.stop_dribble_reward
                components["stop_dribble_reward"][rew_index] = self.stop_dribble_reward

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

    def is_under_pressure(self, observation):
        # Assuming proximity to opponent is a simple proxy for pressure
        return min(np.linalg.norm(observation['right_team'] - observation['left_team'][observation['active']], axis=1)) < self.pressure_threshold

    def is_stopped_dribbling(self, player_index, sticky_actions):
        # Assumes 9 = dribble action is active and checks if it has been stopped
        return sticky_actions[9] == 0 and self.sticky_actions_counter[9] > 0
