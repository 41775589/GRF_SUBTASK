import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive positioning reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions = 0
        self.tackles = 0
        self.interception_reward = 0.3
        self.tackle_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions = 0
        self.tackles = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'interceptions': self.interceptions,
            'tackles': self.tackles
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.interceptions = from_pickle['CheckpointRewardWrapper']['interceptions']
        self.tackles = from_pickle['CheckpointRewardWrapper']['tackles']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "interception_reward": [0.0] * len(reward),
            "tackle_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o['game_mode'] in [2, 5, 6]:  # Defensive modes: GoalKick, ThrowIn, Penalty
                if o['ball_owned_team'] == 0:
                    # Increment interception count if we get the ball in defensive modes
                    self.interceptions += 1
                    components["interception_reward"][rew_index] = self.interception_reward

            if o['sticky_actions'][6] == 1:  # action_bottom
                self.tackles += 1
                components["tackle_reward"][rew_index] = self.tackle_reward

            reward[rew_index] += components["interception_reward"][rew_index] + components["tackle_reward"][rew_index]

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
