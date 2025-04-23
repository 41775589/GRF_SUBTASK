import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on offensive actions aimed at creating scoring opportunities.
    Rewards for Short Pass, Long Pass, Shot, Dribble, and Sprint actions.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions for all agents

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle
    
    def reward(self, reward):
        """
        Enhance the reward based on the quality and frequency of offensive actions.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_actions_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        # Offensive actions coefficients
        coeff_pass_short = 0.1
        coeff_pass_long = 0.15
        coeff_shot = 0.2
        coeff_dribble = 0.05
        coeff_sprint = 0.03

        for rew_index, o in enumerate(observation):
            # Check if any action is taken
            sticky_actions = o['sticky_actions']
            if sticky_actions[1] == 1 or sticky_actions[3] == 1:  # Short Pass or Long Pass
                components["offensive_actions_reward"][rew_index] += coeff_pass_short if sticky_actions[1] == 1 else coeff_pass_long
            if sticky_actions[4] == 1:  # Shot
                components["offensive_actions_reward"][rew_index] += coeff_shot
            if sticky_actions[9] == 1:  # Dribble
                components["offensive_actions_reward"][rew_index] += coeff_dribble
            if sticky_actions[8] == 1:  # Sprint
                components["offensive_actions_reward"][rew_index] += coeff_sprint

            # Update the reward for this index
            reward[rew_index] += components["offensive_actions_reward"][rew_index]

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
