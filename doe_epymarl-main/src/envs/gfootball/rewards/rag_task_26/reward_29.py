import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a midfield dynamic reward focused on role-based contributions."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the midfield tactical regions in terms of x coordinates
        self.midfield_zones = {
            'central': [-0.2, 0.2],  # central midfield zone
            'wide': [-1.0, -0.2, 0.2, 1.0]  # wide areas including left and right midfield
        }
        self.midfield_reward = 0.05  # reward for effective midfield play

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_x = o['left_team'][o['active']][0] if o['active'] >= 0 else None
            
            if player_x is not None:
                if self.midfield_zones['central'][0] <= player_x <= self.midfield_zones['central'][1]:
                    # Rewarding central midfield play
                    components["midfield_reward"][rew_index] += self.midfield_reward
                elif (player_x <= self.midfield_zones['wide'][1] and player_x >= self.midfield_zones['wide'][0]) or \
                     (player_x >= self.midfield_zones['wide'][2] and player_x <= self.midfield_zones['wide'][3]):
                    # Rewarding wide midfield play
                    components["midfield_reward"][rew_index] += self.midfield_reward

                reward[rew_index] += components["midfield_reward"][rew_index]

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
                if action:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
