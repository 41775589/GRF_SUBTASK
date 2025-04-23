import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on the agents' ability to 
    transition effectively between moving and non-moving states as a defensive strategy."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.non_moving_rewards = {}
        self.reward_for_stopping = 0.05  # Reward obtained for effective stop from moving.
        self.movements_threshold = 2  # Movement actions threshold before expecting a stop.

    def reset(self):
        """Reset the environment and clear rewards counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.non_moving_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state for saving along with custom wrapper data."""
        to_pickle['CheckpointRewardWrapper'] = self.non_moving_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state along with custom wrapper data."""
        from_pickle = self.env.set_state(state)
        self.non_moving_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Modify reward based on defensive transitions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defense_transition_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Retrieve and count sticky actions related to movement
            moving_actions = sum(o['sticky_actions'][:7])  # Assuming the first 7 actions relate to movements
            if moving_actions >= self.movements_threshold:
                # Check if a stop has been initialized after meeting the threshold
                if o['sticky_actions'][7] == 1 or o['sticky_actions'][8] == 1:  # assuming actions 7 and 8 are stop/dribble
                    if rew_index not in self.non_moving_rewards:
                        self.non_moving_rewards[rew_index] = 0
                    components["defense_transition_reward"][rew_index] = self.reward_for_stopping
                    reward[rew_index] += components["defense_transition_reward"][rew_index]
                    self.non_moving_rewards[rew_index] += 1
        return reward, components

    def step(self, action):
        """Step function to execute action, get observations, and adjust reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, act in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += act
                info[f"sticky_actions_{i}"] = act
        return observation, reward, done, info
