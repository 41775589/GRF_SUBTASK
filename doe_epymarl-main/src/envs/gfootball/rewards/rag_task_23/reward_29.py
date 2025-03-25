import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive synergy reward focusing on defensive roles and coordination near the penalty area."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize a dictionary to track penalty area control
        self.penalty_area_control = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.penalty_area_control = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['penalty_area_control'] = self.penalty_area_control
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.penalty_area_control = from_pickle['penalty_area_control']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_synergy_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Assuming a defensive scenario near the penalty area: y-coord near zero and x-coord negative
            is_in_penalty_area = -1.0 < o['left_team'][rew_index][0] <= -0.8 and abs(o['left_team'][rew_index][1]) <= 0.2

            # Increase reward for staying in the penalty area with coordination
            if is_in_penalty_area:
                components['defensive_synergy_reward'][rew_index] = 0.05
                # Additional reward if they're actively intercepting or blocking in coordination
                if self.sticky_actions_counter[9] >= 1:  # Assuming index 9 tracks a defensive action like block or intercept
                    components['defensive_synergy_reward'][rew_index] += 0.1

            reward[rew_index] += components['defensive_synergy_reward'][rew_index]

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
