import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific rewards for executing effective stop-dribble control."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions
        # Define custom reward values
        self._dribble_reward = 0.2
        self._stop_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribble_reward": [0.0] * len(reward),
            "stop_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]

            if 'sticky_actions' not in player_obs:
                continue

            # Getting current sticky actions
            current_actions = player_obs['sticky_actions']
            previous_actions = self.sticky_actions_counter

            # Reward for dribbling
            if current_actions[9] == 1 and previous_actions[9] == 0:  # Action dribble
                components["dribble_reward"][rew_index] = self._dribble_reward

            # Reward for stopping to dribble
            if current_actions[9] == 0 and previous_actions[9] == 1:  # Stop dribbling
                components["stop_reward"][rew_index] = self._stop_reward

            # Updating rewards to include new components
            reward[rew_index] += (
                components["dribble_reward"][rew_index] +
                components["stop_reward"][rew_index]
            )

        # Update sticky_actions_counter for the next step
        self.sticky_actions_counter = [o['sticky_actions'] for o in observation]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Include each component into the info dictionary to monitor their effect
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
