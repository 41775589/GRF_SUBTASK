import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for performing high passes and crosses aimed at improving attacking dynamics."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_reward = 0.3
        self.cross_completion_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "cross_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            action_set = o['sticky_actions']
            
            # High Pass
            if action_set[8] == 1 and o['last_action_success'] == True:
                components["pass_reward"][rew_index] = self.pass_completion_reward
                reward[rew_index] += components["pass_reward"][rew_index]

            # Crossing
            if (action_set[9] == 1 and o['ball_owned_team'] == 1 and 
                abs(o['ball'][1]) > 0.30 and o['last_action_success'] == True):
                components["cross_reward"][rew_index] = self.cross_completion_reward
                reward[rew_index] += components["cross_reward"][rew_index]

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
