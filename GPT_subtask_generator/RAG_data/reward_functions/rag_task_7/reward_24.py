import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for successful sliding tackles under high-pressure scenarios."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_count = 0
        self.tackle_attempt_threshold = 5  # Reward every successful 5 tackles
        self.high_pressure_threshold = 0.1  # Distance threshold to determine high pressure situation

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_count = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['tackle_success_count'] = self.tackle_success_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.tackle_success_count = from_pickle['tackle_success_count']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]

            # Determine if a sliding tackle action has just been executed
            if 'sticky_actions' in player_obs and player_obs['sticky_actions'][9] == 1:
                # Identify if the scenario is high pressure: if an opponent is very close
                opponent_distances = np.linalg.norm(
                    player_obs['right_team'] if player_obs['ball_owned_team'] == 0 else player_obs['left_team'],
                    axis=1)
                is_high_pressure = np.any(opponent_distances < self.high_pressure_threshold)

                if is_high_pressure:
                    self.tackle_success_count += 1

                    if self.tackle_success_count >= self.tackle_attempt_threshold:
                        components["tackle_reward"][rew_index] = 1.0
                        self.tackle_success_count = 0

        reward = [sum(x) for x in zip(components["base_score_reward"], components["tackle_reward"])]
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
