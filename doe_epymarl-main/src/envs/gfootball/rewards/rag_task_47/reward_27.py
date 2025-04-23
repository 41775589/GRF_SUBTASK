import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering sliding tackles during counter-attacks in our defensive third."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sliding_tackle_counter = 0
        self.sliding_tackle_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sliding_tackle_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sliding_tackle_counter'] = self.sliding_tackle_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sliding_tackle_counter = from_pickle.get('sliding_tackle_counter', 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sliding_tackle_reward": 0.0}
        if observation is None:
            return reward, components

        for o in observation:
            if 'sticky_actions' in o and o['sticky_actions'][7] == 1 and o['game_mode'] == 0:  # action_bottom_left is a proxy for sliding
                proximity_to_goal = np.abs(o['ball'][0] + 1)  # Since -1 is our goal on x-axis
                if proximity_to_goal < 0.3:  # Defensive third
                    components["sliding_tackle_reward"] += self.sliding_tackle_reward
                    self.sliding_tackle_counter += 1
            reward += components["sliding_tackle_reward"]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = np.sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
