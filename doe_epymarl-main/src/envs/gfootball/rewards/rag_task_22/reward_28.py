import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds sprint usage based reward for defensive coverage improvement."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.initial_positions = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.initial_positions = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'initial_positions': self.initial_positions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        state_from_pickle = self.env.set_state(state)
        custom_state = state_from_pickle['CheckpointRewardWrapper']
        self.sticky_actions_counter = custom_state['sticky_actions_counter']
        self.initial_positions = custom_state['initial_positions']
        return state_from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_usage_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward agents using sprint strategically for repositioning
            sprint_action = o['sticky_actions'][8]  # index 8 corresponds to the sprint action
            if sprint_action:
                self.sticky_actions_counter[rew_index] += 1  # Increment sprint counter

            # Initialize start positions of players for distance calculation
            if self.initial_positions is None:
                self.initial_positions = [player_pos for player_pos in o['left_team']]

            # Calculate distance moved while sprinting and assign reward
            current_position = o['left_team'][rew_index]
            initial_position = self.initial_positions[rew_index]
            distance_moved = np.linalg.norm(np.array(current_position) - np.array(initial_position))

            # Encourage sprinting to quickly change positions defensively
            components['sprint_usage_reward'][rew_index] = distance_moved * 0.05  # Adjust coefficient based on desired influence

            reward[rew_index] += components['sprint_usage_reward'][rew_index]

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
