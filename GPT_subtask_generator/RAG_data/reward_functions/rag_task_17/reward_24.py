import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on high passing and lateral transitions to widen the field of play."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            # Initialize component rewards
            components["pass_accuracy_reward"] = [0.0, 0.0]
            components["position_expansion_reward"] = [0.0, 0.0]

            # Encourage high passes (action 7 in sticky_actions)
            if o['sticky_actions'][7] == 1:
                components["pass_accuracy_reward"][idx] = 0.1
                reward[idx] += components["pass_accuracy_reward"][idx]

            # Reward moving laterally on the field to stretch the defense
            player_x, player_y = o["left_team"][o["active"]][:2] if o['ball_owned_team'] == 0 else o["right_team"][o["active"]][:2]
            lateral_movement = abs(player_y)

            if lateral_movement > 0.3:  # Assuming this is a useful metric to judge lateral positioning
                components["position_expansion_reward"][idx] = 0.05
                reward[idx] += components["position_expansion_reward"][idx]

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
