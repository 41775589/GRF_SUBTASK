import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the reward for exhibiting skills related to ball control and effective passing
    under pressure in tight game situations, focusing on Short Pass, High Pass, and Long Pass.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Counter for sticky actions usage.
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize reward components multipliers
        self.short_pass_reward = 0.2
        self.high_pass_reward = 0.3
        self.long_pass_reward = 0.5

    def reset(self):
        # Reset sticky actions counter on each episode
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
        
    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
        
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle
    
    def reward(self, reward):
        """
        Enhances the reward based on effective passing actions under tight situations.
        """
        components = {"base_score_reward": reward.copy(),
                      "short_pass_reward": [0.0] * len(reward),
                      "high_pass_reward": [0.0] * len(reward),
                      "long_pass_reward": [0.0] * len(reward)}
        
        if self.env.unwrapped.observation() is None:
            return reward, components

        observation = self.env.unwrapped.observation()
        
        for idx in range(len(reward)):
            o = observation[idx]
            own_pos = o['left_team'][o['active']]
            opponents = o['right_team']

            # Calculate distances to all opponents to determine pressure
            pressure = np.min(np.linalg.norm(opponents - own_pos, axis=1))

            if pressure < 0.1: # assuming under high-pressure if opponents are very close
                # Check sticky actions that represent passing
                if o['sticky_actions'][0] == 1:  # short pass
                    reward[idx] += self.short_pass_reward
                    components["short_pass_reward"][idx] = self.short_pass_reward
                if o['sticky_actions'][1] == 1:  # high pass
                    reward[idx] += self.high_pass_reward
                    components["high_pass_reward"][idx] = self.high_pass_reward
                if o['sticky_actions'][2] == 1:  # long pass
                    reward[idx] += self.long_pass_reward
                    components["long_pass_reward"][idx] = self.long_pass_reward

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
                if i < 10:
                    self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
