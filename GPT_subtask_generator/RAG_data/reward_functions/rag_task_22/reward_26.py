import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides a reward for effective sprinting and defensive positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._sprint_reward = 0.05
        self._positioning_reward = 0.1
        self.sprint_threshold = 3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'StickyActions': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['StickyActions'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for i in range(len(reward)):
            o = observation[i]
            spr_actions = o['sticky_actions'][8]  # Sprint action index is 8

            # Reward for consistent sprinting
            if spr_actions == 1:
                self.sticky_actions_counter[i] += 1
            else:
                self.sticky_actions_counter[i] = 0
                
            if self.sticky_actions_counter[i] >= self.sprint_threshold:
                components["sprint_reward"][i] = self._sprint_reward
                reward[i] += components["sprint_reward"][i]
            
            # Reward for moving into strategically important positions
            # Assuming the agents are on the left team, boost rewards when they move towards the right (opponent's side)
            # Prioritize the y-positions closer to midfield (y ~ 0 in normalized environment coordinates)
            if o['left_team'][o['active']][0] > 0 and abs(o['left_team'][o['active']][1]) < 0.15:
                components["positioning_reward"][i] = self._positioning_reward
                reward[i] += components["positioning_reward"][i]
                
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
