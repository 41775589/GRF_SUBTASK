import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for dribbling and sprinting skills enhancement."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the sticky actions counter on environment reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serializes the state including custom sticky actions state."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserializes the state including custom sticky actions state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10))
        return from_pickle

    def reward(self, reward):
        """Compute reward by considering dribbling and sprinting achievements."""
        observation = self.env.unwrapped.observation()
        o = observation if observation else {}
        components = {"base_score_reward": reward.copy(), "sprint_dribble_bonus": 0.0}

        if o and 'sticky_actions' in o:
            sprint_active = o['sticky_actions'][8]  # Index 8 relates to 'action_sprint'
            dribble_active = o['sticky_actions'][9]  # Index 9 relates to 'action_dribble'

            # Increment counters if actions are active
            if sprint_active:
                self.sticky_actions_counter[8] += 1

            if dribble_active:
                self.sticky_actions_counter[9] += 1

            # Give a reward based on the activation frequency and the correct execution.
            components['sprint_dribble_bonus'] = 0.05 * self.sticky_actions_counter[8] + 0.05 * self.sticky_actions_counter[9]
            reward += components['sprint_dribble_bonus']

        return reward, components

    def step(self, action):
        """Applies the action, computes the rewards and returns results."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
