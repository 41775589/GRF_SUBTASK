import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful standing tackles during the game."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.successful_tackles = 0
        self._tackle_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.successful_tackles = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['successful_tackles'] = self.successful_tackles
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.successful_tackles = from_pickle.get('successful_tackles', 0)
        return from_pickle

    def reward(self, reward):
        """Reward is given for successful tackles, which increase possession without fouls."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if self.is_successful_tackle(o):
                components["tackle_reward"][rew_index] = self._tackle_reward
                reward[rew_index] += components["tackle_reward"][rew_index]
                self.successful_tackles += 1

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

    def is_successful_tackle(self, observation):
        """Check if a tackle was successful considering the environment's observation."""
        return observation['game_mode'] == 0 and observation['ball_owned_team'] == 1 and \
               "tackle" in observation['sticky_actions']
