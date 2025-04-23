import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful sliding tackles near the defensive third."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_counter = 0
        self._tackle_zone_threshold = -0.3  # Represent X threshold for defensive third

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['tackle_success_counter'] = self.tackle_success_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.tackle_success_counter = from_pickle['tackle_success_counter']
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
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Check for sliding tackle action
            is_sliding_tackle = o['sticky_actions'][6]  # Assuming index 6 corresponds to sliding tackle

            if 'left_team' in o:
                for player in o['left_team']:
                    player_x, _ = player
                    # Check if tackle in the right zone
                    if is_sliding_tackle and player_x < self._tackle_zone_threshold:
                        components["tackle_reward"][rew_index] = 2.0  # Reward for a successful tackle
                        reward[rew_index] += components["tackle_reward"][rew_index]
                        self.tackle_success_counter += 1

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
