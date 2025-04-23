import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that augments the reward function to encourage agents to learn offensive skills including
    passing, shooting, dribbling, and creating scoring opportunities.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_coefficient = 0.1
        self.shot_coefficient = 0.2
        self.dribble_coefficient = 0.05
        self.sprint_coefficient = 0.03

    def reset(self):
        """
        Reset the environment and associated variables for sticky action counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state of the environment.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Load the state of the environment.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward based on the actions performed by the agents, focused on offensive actions.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "action_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]

            last_actions = self.sticky_actions_counter

            # Check and add reward for the actions
            if o['sticky_actions'][5]:  # Shot action index
                components['action_reward'][i] += self.shot_coefficient

            if o['sticky_actions'][0] or o['sticky_actions'][1]:   # Pass action indices (short and long)
                components['action_reward'][i] += self.pass_coefficient

            if o['sticky_actions'][9]:  # Dribble action index
                components['action_reward'][i] += self.dribble_coefficient

            if o['sticky_actions'][8]:  # Sprint action index
                components['action_reward'][i] += self.sprint_coefficient

            # Calculate final reward
            reward[i] += components['action_reward'][i]

        return reward, components

    def step(self, action):
        """
        Steps through the environment, applying the modified reward function.
        """
        observation, reward, done, info = self.env.step(action)
        self.sticky_actions_counter += observation['sticky_actions']
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update info for detailed debug
        for i in range(10):
            info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
