import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive actions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "defensive_actions_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            sliding_tackle = o['sticky_actions'][3]  # Assuming index 3 for sliding tackle action
            intercept_kick = o['sticky_actions'][2]  # Assuming index 2 for intercepting or blocking kicks
            aggressive = o['sticky_actions'][1]      # Additional aggressive defense action if needed

            # Check if the player's team does not possess the ball
            if o['ball_owned_team'] == 1 or o['ball_owned_team'] == -1:
                # Reward negative of sticky actions if the player is involved in defensive actions
                # Here we assume defensive actions are valuable and hinder opponent's progress
                components['defensive_actions_reward'][rew_index] = (
                    0.2 * sliding_tackle + 0.1 * intercept_kick + 0.05 * aggressive
                )
                reward[rew_index] += components['defensive_actions_reward'][rew_index]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
