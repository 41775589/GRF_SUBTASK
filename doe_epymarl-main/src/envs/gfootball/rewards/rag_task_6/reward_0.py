import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effective energy management using Stop-Sprint and Stop-Moving actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.was_sprinting = False

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.was_sprinting = False
        return self.env.reset()

    def reward(self, reward):
        """Modify the rewards given based on the usage of sprint and movement controls to encourage stamina conservation."""
        observation = self.env.unwrapped.observation()

        for rew_index, o in enumerate(observation):
            # Determine sprint and movement actions from sticky actions.
            is_sprinting = bool(o['sticky_actions'][8])  # Sprint action index is 8
            is_moving = any(o['sticky_actions'][:8])  # Movement actions are the first 8
            
            if self.was_sprinting and not is_sprinting:
                # Reward stopping sprinting when not required
                reward[rew_index] += 0.1
            
            if not is_moving and not is_sprinting:
                # Reward staying idle, not depleting stamina unnecessarily
                reward[rew_index] += 0.05
            
            # Track the sprint state for the next step
            self.was_sprinting = is_sprinting

        components = {
            "base_score_reward": reward.copy(),
            "stop_sprint_reward": [0.1 if self.was_sprinting and not is_sprinting else 0.0 for is_sprinting in 
                                   [agent['sticky_actions'][8] for agent in observation]],
            "stop_moving_reward": [0.05 if not agent['sticky_actions'][:8].any() and not agent['sticky_actions'][8]
                                   else 0.0 for agent in observation]
        }

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
        return observation, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['was_sprinting'] = self.was_sprinting
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.was_sprinting = from_pickle['was_sprinting']
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle
