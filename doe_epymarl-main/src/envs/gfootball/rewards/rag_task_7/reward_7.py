import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering defensive maneuvers, especially sliding tackles."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Resets the environment and sticky actions counter for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Return the environment state with additional sliding tackle count."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the sliding tackle count and environment state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Calculate augmented reward based on successful defensive maneuvers."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "tackle_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_action = o['sticky_actions']

            # Reward agents for the successful sliding tackle (action index for slide tackle assumed to be 3)
            if current_action[3] == 1 and o['ball_owned_team'] == 1:  # Assuming ball_owned_team 1 is opponent
                components["tackle_reward"][rew_index] = 0.5  # Reward for successful tackle
                reward[rew_index] += components["tackle_reward"][rew_index]
                self.sticky_actions_counter[3] += 1  # Increment tackle count

        return reward, components

    def step(self, action):
        """Perform environment step and apply the augmented reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        for rew_index in range(len(reward)):
            for i in range(10):
                info[f"sticky_actions_{rew_index}_{i}"] = observation[rew_index]['sticky_actions'][i]
        
        return observation, reward, done, info
