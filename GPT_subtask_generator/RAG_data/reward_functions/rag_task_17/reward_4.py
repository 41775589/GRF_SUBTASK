import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on mastering midfield-wide positioning, high passes, and aiding in stretching the opposition."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.start_position_rewards = {}
        # To encourage spreading out and controlling the ball in wide areas
        self.wide_field_bonus = 0.1
        self.high_pass_bonus = 0.2
        # Initialize sticky actions counter (10 potential sticky actions)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.start_position_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.start_position_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.start_position_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "wide_field_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Prefer agents in wider positions but in adversary's half
            if o['right_team'][o['active']][0] > 0:  # X position > 0, adversary half
                y_pos = abs(o['right_team'][o['active']][1])  # Get the absolute Y pos
                if y_pos > 0.2:  # Considered "wide" if Y is greater than 0.2
                    components["wide_field_reward"][rew_index] = self.wide_field_bonus
                    reward[rew_index] += components["wide_field_reward"][rew_index]
            
            # Reward for performing high passes
            if 'sticky_actions' in o:
                if o['sticky_actions'][football_action_set.action_high_pass]:
                    components["wide_field_reward"][rew_index] += self.high_pass_bonus
                    reward[rew_index] += components["wide_field_reward"][rew_index]

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
