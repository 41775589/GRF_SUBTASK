import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive posture reward supporting quick transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.active_change_reward = 0.5  # Reward for changing active player effectively
        self.decreasing_cooldown = 10  # Cooldown steps for action switching

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
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0] * len(reward)}

        for i, (rew, obs) in enumerate(zip(reward, observation)):
            # Tracks changes in active player to encourage dynamic defenses
            current_active = obs['active']
            previous_active = 'previously_active' in obs and obs['previously_active']
            active_changed = current_active != previous_active

            # Encourage changing active player defensively and efficiently
            if active_changed and self.sticky_actions_counter[current_active] < self.decreasing_cooldown:
                components['transition_reward'][i] = self.active_change_reward
                reward[i] += components['transition_reward'][i]
                self.sticky_actions_counter[current_active] = 0
            else:
                self.sticky_actions_counter[current_active] += 1

            # Remember the active player for the next step's comparison
            obs['previously_active'] = current_active

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Aggregate reward information for diagnostics
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Count how many actions have reached transitions cooldown to zero
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
