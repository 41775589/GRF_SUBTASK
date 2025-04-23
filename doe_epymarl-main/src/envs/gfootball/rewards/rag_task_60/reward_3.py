import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive transitions reward based on stopping and starting movements."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_reward_coef = 0.05  # Change this coefficient to scale the transition rewards

    def reset(self):
        """
        Reset the environment and clear the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """
        Save the current state.
        """
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the environment state from a saved state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle

    def reward(self, reward):
        """
        Calculate the reward for the defensive transitioning based on the observed actions and sticky states.
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'transition_reward': []}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation), "The reward and observation lengths must match."

        # Loop through each agent's observation
        for i, o in enumerate(observation):
            trans_reward = 0
            # Check if the active player just stopped (action might have been 0 after being non-zero in the previous step)
            if o['sticky_actions'][0] == 0 and self.sticky_actions_counter[0] > 0:
                trans_reward += self.transition_reward_coef

            # Check if the active player just started moving (previous action was 0 and current is non-zero)
            for j in range(10):
                if o['sticky_actions'][j] and self.sticky_actions_counter[j] == 0:
                    trans_reward += self.transition_reward_coef
            
            # Update the current sticky actions to the observation
            self.sticky_actions_counter = o['sticky_actions']

            components['transition_reward'].append(trans_reward)
            reward[i] += trans_reward

        return reward, components

    def step(self, action):
        """
        Perform a step in the environment with the given action, calculate the modified reward,
        and return the results.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        # Additional logging of reward components and final reward
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        return observation, reward, done, info
