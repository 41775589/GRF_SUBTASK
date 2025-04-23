import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive skills and counterattacks through added rewards."""

    def __init__(self, env):
        super().__init__(env)
        self.positions_counter = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the positions counter and sticky actions counter on episode start."""
        self.positions_counter = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Saves the state of the checkpoint rewards."""
        to_pickle['positions_counter'] = self.positions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the state of the checkpoint rewards."""
        from_pickle = self.env.set_state(state)
        self.positions_counter = from_pickle['positions_counter']
        return from_pickle

    def reward(self, reward):
        """Rewards for strategic positioning and quick transitions for defensive strategies."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        for rew_index, o in enumerate(reward):
            components[f"position_{rew_index}"] = 0.0
            # Check if the agent should be rewarded for defensive actions
            if 'active' in observation and 'left_team' in observation:
                # Reward players for moving into strategic defensive or counterattack positions
                player_pos = observation['left_team'][observation['active']]
                if player_pos[0] < -0.5:  # Defensive positions in own half
                    if rew_index not in self.positions_counter:
                        self.positions_counter[rew_index] = {0: False}
                    if not self.positions_counter[rew_index][0]:
                        components[f"position_{rew_index}"] = 0.1  # Reward for first defensive position
                        self.positions_counter[rew_index][0] = True
                elif player_pos[0] > 0.5:  # Positions favorable for counterattacks
                    if rew_index not in self.positions_counter:
                        self.positions_counter[rew_index] = {1: False}
                    if not self.positions_counter[rew_index][1]:
                        components[f"position_{rew_index}"] = 0.2  # Higher reward for counterattack positions
                        self.positions_counter[rew_index][1] = True
            reward[rew_index] += components[f"position_{rew_index}"]

        return reward, components

    def step(self, action):
        """Steps through environment, using modified rewards from the wrapper."""
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
