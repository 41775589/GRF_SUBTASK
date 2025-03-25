import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive strategies such as shooting, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        state = self.env.set_state(state)
        self.sticky_actions_counter = state.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Reward for goal scoring actions
            if o['game_mode'] == 6 and o['ball_owned_team'] == o['active']:
                components['shooting_reward'][rew_index] = 1.0
                reward[rew_index] += 1.0

            # Reward for dribbling (keeping possession under pressure)
            if o['sticky_actions'][9] == 1 and o['ball_owned_team'] == 0:
                components['dribbling_reward'][rew_index] = 0.5
                reward[rew_index] += 0.5

            # Reward for successful passes (change of ball ownership among teammates)
            if self.sticky_actions_counter[8] > 0 and o['ball_owned_team'] == 0:
                components['passing_reward'][rew_index] = 0.2
                reward[rew_index] += 0.2

            # Update sticky actions counters
            self.sticky_actions_counter = o['sticky_actions']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
