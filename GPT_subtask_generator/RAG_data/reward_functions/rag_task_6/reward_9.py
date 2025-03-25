import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages learning stamina conservation using Stop-Sprint 
       and Stop-Moving actions, crucial for maintaining stamina and positional 
       integrity over the duration of a match."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Note: Explicitly define rewarding for relevant actions.
        # action indices based on IMPORTED action set configuration.
        # index 8 corresponds to sprint, 0 to move left, which are critical for simulating energy conservation.
        self.sprint_action_index = 8  
        self.move_actions_indexes = [0, 1, 2, 3, 4, 5, 6]  # These are hypothetical indices for movement-related actions.
        self.non_move_and_sprint_action_reward = 0.005  # Reward added for avoiding unnecessary movement/sprinting.
        self.tired_threshold = 0.1  # Threshold to consider a player as tired.

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
        components = {
            "base_score_reward": reward.copy(),
            "stamina_preservation_reward": np.zeros_like(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check for tiredness
            if ('right_team_tired_factor' in o and o['active'] >= 0 and
                o['right_team_tired_factor'][o['active']] > self.tired_threshold):
                # Check if non-movement/sprint actions
                # Assuming sprint is in index 8 and moving actions are in indices 0-6
                if not any(o['sticky_actions'][idx] for idx in self.move_actions_indexes + [self.sprint_action_index]):
                    components['stamina_preservation_reward'][rew_index] = self.non_move_and_sprint_action_reward
                    reward[rew_index] += components['stamina_preservation_reward'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
