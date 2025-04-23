import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides rewards for accuracy and precision in making long passes across the football field.
    It specifically targets reward based on successful long passes over different lengths, 
    simulating various match conditions like player positions and ball dynamics.
    """

    def __init__(self, env):
        super().__init__(env)
        self.min_pass_distance = 0.3  # Minimal distance to consider a pass as 'long'
        self.max_pass_distance = 1.0  # Maximum normalized distance across the field
        self.accuracy_reward = 1.0  # Reward for the accuracy in reaching the player
        self.long_pass_bonus = 0.5  # Bonus reward for achieving max distance threshold
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_accuracy": [0.0] * len(reward),
                      "long_pass_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index, obs in enumerate(observation):
            dist = np.linalg.norm(obs['ball_direction'][:2])
            if dist >= self.min_pass_distance:
                reward[rew_index] += self.accuracy_reward
                components["long_pass_accuracy"][rew_index] = self.accuracy_reward

                if dist >= self.max_pass_distance:
                    reward[rew_index] += self.long_pass_bonus
                    components["long_pass_bonus"][rew_index] = self.long_pass_bonus
                    
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
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
