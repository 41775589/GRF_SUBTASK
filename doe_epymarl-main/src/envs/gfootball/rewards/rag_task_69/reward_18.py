import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that augments the reward for offensive strategies: accurate shooting,
       effective dribbling to evade opponents, and different types of passing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_reward = 0.5
        self.dribbling_reward = 0.3
        self.passing_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_types = {
            'long': 0.15,
            'high': 0.05
        }

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

            # Shooting reward
            if o['game_mode'] == 6:  # Penalty game mode for shooting
                components["shooting_reward"][rew_index] = self.shooting_reward
                reward[rew_index] += components["shooting_reward"][rew_index]

            # Dribbling reward
            if o['sticky_actions'][9] == 1:  # Checking if dribbling action is active
                components["dribbling_reward"][rew_index] += self.dribbling_reward
                reward[rew_index] += components["dribbling_reward"][rew_index]

            # Passing reward calculation based on pass type (simulated here as different game modes)
            if o['game_mode'] == 5:  # Assume game_mode 5 is for long passes
                components["passing_reward"][rew_index] += self.pass_types['long']
                reward[rew_index] += components["passing_reward"][rew_index]
            if o['game_mode'] == 4:  # Assume game_mode 4 is for high passes
                components["passing_reward"][rew_index] += self.pass_types['high']
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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
