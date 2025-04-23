import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds custom rewards for performing sliding tackles effectively,
    particularly during defensive counter-attacks.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.near_defensive_third_counter = {}
        self.sliding_tackle_counter = {}
        self.defensive_positioning_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.near_defensive_third_counter = {}
        self.sliding_tackle_counter = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'near_defensive_third_counter': self.near_defensive_third_counter,
            'sliding_tackle_counter': self.sliding_tackle_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.near_defensive_third_counter = from_pickle['CheckpointRewardWrapper']['near_defensive_third_counter']
        self.sliding_tackle_counter = from_pickle['CheckpointRewardWrapper']['sliding_tackle_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            is_near_defensive_third = o['left_team'][:, 0].min() > -0.5

            # Check if the agent is in the defensive third and record it
            if is_near_defensive_third:
                if rew_index not in self.near_defensive_third_counter:
                    self.near_defensive_third_counter[rew_index] = True
                    components["defensive_positioning_reward"][rew_index] += self.defensive_positioning_reward
            
            # Check for sliding tackle action (action index for slide is specific, e.g., 4)
            slid_action = o['sticky_actions'][4] == 1
            if slid_action and is_near_defensive_third:
                if rew_index not in self.sliding_tackle_counter:
                    self.sliding_tackle_counter[rew_index] = True
                    components["defensive_positioning_reward"][rew_index] += self.defensive_positioning_reward

            # Update the reward based on the components
            reward[rew_index] += components["defensive_positioning_reward"][rew_index]

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
