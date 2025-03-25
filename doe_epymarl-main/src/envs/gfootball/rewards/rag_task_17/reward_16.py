import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on high passing accuracy and good lateral positioning."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._high_pass_accuracy_reward = 0.1 
        self._lateral_positioning_reward = 0.1
        self.high_passes = {}
        self.good_positioning = {}
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_passes = {}
        self.good_positioning = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['high_passes'] = self.high_passes
        to_pickle['good_positioning'] = self.good_positioning
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.high_passes = from_pickle.get('high_passes', {})
        self.good_positioning = from_pickle.get('good_positioning', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_accuracy_reward": [0.0] * len(reward),
            "lateral_positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_reward_base = components["base_score_reward"][rew_index]
            
            # Detect successful high passes
            if 'high_pass' in o['sticky_actions'] and o['sticky_actions']['high_pass']:
                if rew_index not in self.high_passes:
                    components["high_pass_accuracy_reward"][rew_index] = self._high_pass_accuracy_reward
                    reward[rew_index] += components["high_pass_accuracy_reward"][rew_index]
                    self.high_passes[rew_index] = True

            # Reward for maintaining width in possession
            x_position = abs(o['ball'][0])  # normalize x position
            if x_position > 0.7:  # assuming positions are on a scale from -1 to 1
                if rew_index not in self.good_positioning:
                    components["lateral_positioning_reward"][rew_index] = self._lateral_positioning_reward
                    reward[rew_index] += components["lateral_positioning_reward"][rew_index]
                    self.good_positioning[rew_index] = True

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        new_reward, components = self.reward(reward)
        info["final_reward"] = sum(new_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, new_reward, done, info
