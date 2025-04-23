import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering sliding tackles during high-pressure defensive scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sliding_tackle_reward = 0.5
        self.high_pressure_zone = -0.5  # Defensive third boundary for the left side
        self.last_ball_owner = None
        self.tackle_counter = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_owner = None
        self.tackle_counter = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        to_pickle['ball_owner'] = self.last_ball_owner
        to_pickle['tackle_counter'] = self.tackle_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        self.last_ball_owner = from_pickle['ball_owner']
        self.tackle_counter = from_pickle['tackle_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_owner = (o['ball_owned_team'], o['ball_owned_player'])

            # Check if in high-pressure zone and if action was a slide
            if o['left_team'][o['active']][0] <= self.high_pressure_zone:
                if o['sticky_actions'][9] == 1:  # Index 9 corresponds to the "sliding" action
                    if self.last_ball_owner == (1, o['active']):  # Ball was owned by opposing player
                        if rew_index not in self.tackle_counter:
                            self.tackle_counter[rew_index] = 0
                        if self.tackle_counter[rew_index] < 3:  # Limit rewards per episode
                            components["tackle_reward"][rew_index] = self.sliding_tackle_reward
                            reward[rew_index] += self.sliding_tackle_reward
                            self.tackle_counter[rew_index] += 1

            self.last_ball_owner = current_owner

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
