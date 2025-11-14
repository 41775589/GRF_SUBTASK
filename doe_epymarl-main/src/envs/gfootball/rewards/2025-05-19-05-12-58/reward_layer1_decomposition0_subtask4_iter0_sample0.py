import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward focused on mastering Sliding for tackles and
    reactively using Stop-Moving and Stop-Sprint to effectively engage and neutralize attackers."""

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
        """Custom reward logic focusing on sliding tackles, stop moving, and stop sprint engages."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "tackle_reward": [0.0] * len(reward)}

        if not observation:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check if there's active defensive play.
            sliding_action = o['sticky_actions'][9]  # Assuming index 9 in sticky_actions is sliding.
            stop_moving_action = o['sticky_actions'][6]  # Assuming index 6 in sticky_actions is stop-moving.
            stop_sprint_action = o['sticky_actions'][8]  # Assuming index 8 in sticky_actions is stop-sprint.

            # Provide extra reward for successful sliding action and stopping under appropriate conditions.
            if sliding_action or (stop_moving_action and stop_sprint_action):
                components['tackle_reward'][rew_index] = 0.5

            # Bonus for a sequence leading to neutralizing an attacker.
            if sliding_action and stop_moving_action and stop_sprint_action:
                components['tackle_reward'][rew_index] += 1.0

            reward[rew_index] += components['tackle_reward'][rew_index]

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
