import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for football offensive strategies:
       accurate shooting, effective dribbling, and strategic passing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward coefficients
        self.shooting_reward = 1.0
        self.dribbling_reward = 0.3
        self.passing_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            prev_score = components["base_score_reward"][rew_index]

            # Check if a goal was scored
            if o['score'][0] > o['score'][1]:  # improved goal scoring assumption
                components["shooting_reward"][rew_index] = self.shooting_reward
                reward[rew_index] += components["shooting_reward"][rew_index]

            # Check for effective dribbling (ball ownership with movement)
            if o['ball_owned_team'] == 0 and o['sticky_actions'][9]:  # Action dribble is active
                components["dribbling_reward"][rew_index] = self.dribbling_reward
                reward[rew_index] += components["dribbling_reward"][rew_index]

            # Check for strategic passes (change in ball ownership)
            if o['ball_owned_team'] == 0 and o['sticky_actions'][1] in [2, 3]:  # Pass high or long
                components["passing_reward"][rew_index] = self.passing_reward
                reward[rew_index] += components["passing_reward"][rew_index]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
