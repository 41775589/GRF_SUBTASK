import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ Wrapper that adds rewards for defensive actions and positioning. """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reward_for_tackle = 0.2
        self.reward_for_sliding = 0.3
        self.position_reward_factor = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        state = self.env.set_state(state)
        self.sticky_actions_counter = state['CheckpointRewardWrapper']
        return state

    def reward(self, reward):
        """ Adjust the reward based on defensive actions and positioning. """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "sliding_reward": [0.0] * len(reward),
                      "position_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for tackling
            if o['sticky_actions'][3]:  # Assuming index 3 is tackle action
                components["tackle_reward"][rew_index] = self.reward_for_tackle

            # Reward for sliding
            if o['sticky_actions'][4]:  # Assuming index 4 is slide action
                components["sliding_reward"][rew_index] = self.reward_for_sliding

            # Reward for defensive positioning
            # Assuming that a good position is closer to own goal (e.g., x < -0.5)
            if o['left_team_roles'][o['active']] == 1 and o['ball'][0] < -0.5:  # if player is a defender and in own half
                components["position_reward"][rew_index] = self.position_reward_factor * (1 + np.abs(o['ball'][0]))
            
            reward[rew_index] += sum(components[key][rew_index] for key in components)

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
