import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds additional rewards for successful sliding tackles during defensive scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_counter = 0
        self.prev_ball_owned_team = -1
        self.tackle_reward = 0.5  # Reward value for a successful tackle

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_counter = 0
        self.prev_ball_owned_team = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['tackle_success_counter'] = self.tackle_success_counter
        to_pickle['prev_ball_owned_team'] = self.prev_ball_owned_team
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.tackle_success_counter = from_pickle['tackle_success_counter']
        self.prev_ball_owned_team = from_pickle['prev_ball_owned_team']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 1 and self.prev_ball_owned_team == 0 and o['game_mode'] in {0, 2, 3, 4, 5, 6}:
                # If ball was previously owned by our team, but now it's owned by the opponent
                components["tackle_reward"][rew_index] = self.tackle_reward
                reward[rew_index] += components["tackle_reward"][rew_index]
                self.tackle_success_counter += 1

        self.prev_ball_owned_team = observation[0]['ball_owned_team']  # Assumes consistency across observations
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
