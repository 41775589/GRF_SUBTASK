import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for performing quick attacks and adapting during various game modes."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.quick_attack_bonus = 0.2
        self.adaptation_bonus = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        components = {"base_score_reward": reward.copy(),
                      "quick_attack_bonus": [0.0] * len(reward),
                      "adaptation_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward quick offensive play when possession changes quickly towards the goal
            if o['game_mode'] in [1, 3, 5]:  # Modes 1, 3, 5 relate to dynamic game states like kickoffs and throw-ins
                components["quick_attack_bonus"][rew_index] = self.quick_attack_bonus
                reward[rew_index] += components["quick_attack_bonus"][rew_index]

            # Reward adaptation to change in game phase
            if o['game_mode'] != self.env.unwrapped.previous_game_mode:
                components["adaptation_bonus"][rew_index] = self.adaptation_bonus
                reward[rew_index] += components["adaptation_bonus"][rew_index]

            self.env.unwrapped.previous_game_mode = o['game_mode']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Track sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
