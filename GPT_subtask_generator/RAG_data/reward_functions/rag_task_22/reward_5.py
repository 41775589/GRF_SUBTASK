import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes sprint usage for faster defensive positioning."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_usage_counter = np.zeros(2, dtype=int)  # Assuming two agents for simplicity

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_usage_counter = np.zeros(2, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sprint_usage_counter'] = self.sprint_usage_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sprint_usage_counter = from_pickle.get('sprint_usage_counter', np.zeros(2, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Sprint action is index 8 in sticky_actions
            if o['sticky_actions'][8]:  # Check if sprint action is active
                self.sprint_usage_counter[rew_index] += 1

            # Incremental reward for using sprint - incentivizes using sprint often
            components["sprint_reward"][rew_index] = 0.1 * self.sprint_usage_counter[rew_index]
            reward[rew_index] += components["sprint_reward"][rew_index]
        
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
