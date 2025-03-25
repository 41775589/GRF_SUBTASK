import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive tactical reward designed to improve defensive skills, such as responsiveness and interception."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_interceptions = 0
        self._interception_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_interceptions = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['interceptions'] = self._num_interceptions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._num_interceptions = from_pickle['interceptions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            prev_ball_owned_team = o['prev_ball_owned_team']
            ball_owned_team = o['ball_owned_team']
            active_player_team = o['active'] in o['left_team']

            if prev_ball_owned_team == 1 and ball_owned_team == 0:
                # Check if the active player belongs to the defending team and contributed to interception
                if active_player_team:
                    components["defensive_reward"][rew_index] = self._interception_reward
                    reward[rew_index] += components["defensive_reward"][rew_index]
                    self._num_interceptions += 1

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
