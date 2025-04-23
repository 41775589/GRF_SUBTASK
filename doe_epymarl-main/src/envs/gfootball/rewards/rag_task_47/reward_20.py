import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a task-specific reward for mastering sliding tackles during counter-attacks and high-pressure situations near our defensive third."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._sliding_tackle_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "sliding_tackle": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        # Loop through observations for each player/agent
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check for defensive third sliding tackles
            if o['ball'][0] < 0 and o['sticky_actions'][9] == 1:  # Assuming index 9 is the sliding tackle action
                if (o['ball_owned_team'] == 1 and  # Ball owned by the opponent in the defensive third
                    o['left_team'][o['active']][0] < 0.2):  # Active player close to our goal
                    reward[rew_index] += self._sliding_tackle_reward
                    components["sliding_tackle"][rew_index] = self._sliding_tackle_reward

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
