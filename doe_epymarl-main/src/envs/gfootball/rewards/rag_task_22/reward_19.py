import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a sprint-reward for improving defensive coverage."""

    def __init__(self, env):
        super().__init__(env)
        # Parameters for sprint rewarding
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.max_x_position = 0.0  # Keeps track of the furthest right x position

    def reset(self):
        """Reset for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.max_x_position = 0.0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the wrapper."""
        to_pickle['CheckpointRewardWrapper'] = self.max_x_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the wrapper from a pickle item."""
        from_pickle = self.env.set_state(state)
        self.max_x_position = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Calculate the reward with considerations for defensive sprint efficiency."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'right_team_active' in o and o['right_team_active']:
                # Get x-coordinates of active players
                x_positions = o['right_team'][:, 0]
                max_current = np.max(x_positions)
                sprint_action_active = o['sticky_actions'][8]  # Index 8 is sprint action

                # Encourage moving rapidly to positions further down the field to form defense
                if sprint_action_active and max_current > self.max_x_position:
                    components['sprint_reward'][rew_index] = 0.1 * (max_current - self.max_x_position)
                else:
                    components['sprint_reward'][rew_index] = 0.0
                
                # Update maximum position reached
                self.max_x_position = max(self.max_x_position, max_current)
                reward[rew_index] += components['sprint_reward'][rew_index]

        return reward, components

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return ob, reward, done, info
