import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for energy conservation with Stop-Sprint and Stop-Moving actions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.prev_actions = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stamina_maintain_reward = 0.05

    def reset(self):
        """
        Reset the environment and turn off all sticky actions.
        """
        self.prev_actions = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save state with additional data.
        """
        to_pickle['CheckpointRewardWrapper'] = self.prev_actions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Load previous state and additional data.
        """
        from_pickle = self.env.set_state(state)
        self.prev_actions = from_pickle.get('CheckpointRewardWrapper', {})
        return from_pickle

    def reward(self, reward):
        """
        Modify the base reward based on the efficient usage of sprint and moving actions.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stamina_maintain_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sticky_actions = o['sticky_actions']
            prev_sticky = self.prev_actions.get(rew_index, np.zeros_like(sticky_actions))

            # Reward for stopping sprint (action 8) and any movement (actions 0-7)
            if (prev_sticky[8] == 1 and sticky_actions[8] == 0):
                # Reward for stopping sprint
                components["stamina_maintain_reward"][rew_index] += self.stamina_maintain_reward
                reward[rew_index] += components["stamina_maintain_reward"][rew_index]

            # Check if all movement actions are stopped
            if np.any(prev_sticky[:8] == 1) and not np.any(sticky_actions[:8] == 1):
                # Reward for stopping movement
                components["stamina_maintain_reward"][rew_index] += self.stamina_maintain_reward
                reward[rew_index] += components["stamina_maintain_reward"][rew_index]

            # Update previous actions for the next reward calculation
            self.prev_actions[rew_index] = sticky_actions.copy()

        return reward, components

    def step(self, action):
        """
        Step function should remain standard apart from gathering rewards, which are parsed through the custom reward function.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    if action == 1:
                        self.sticky_actions_counter[i] += 1 
                        info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
