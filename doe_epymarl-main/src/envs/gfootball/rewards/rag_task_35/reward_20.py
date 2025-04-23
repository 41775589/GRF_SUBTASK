import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward for maintaining strategic positioning and effective transitions between defense and attack."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards = np.zeros((2, 11))  # Rewards for 11 strategic positions
        self.transition_reward = 0.2  # Reward for effective transitions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.position_rewards.fill(0.0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'position_rewards': self.position_rewards}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.position_rewards = from_pickle['CheckpointRewardWrapper']['position_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "position_rewards": np.zeros_like(reward),
                      "transition_rewards": np.zeros_like(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            # Reward strategic positioning
            for j in range(11):
                if self.is_in_strategic_position(o, j):
                    if self.position_rewards[i, j] == 0.0:
                        self.position_rewards[i, j] = 1.0
                        components["position_rewards"][i] += 0.1
            
            # Reward transitions between positions
            if self.successful_transition(o):
                components["transition_rewards"][i] += self.transition_reward
            
            reward[i] += components["position_rewards"][i] + components["transition_rewards"][i]

        return reward, components

    def is_in_strategic_position(self, observation, position_index):
        # Check if agent is in one of the strategic positions
        # Dummy function, replace with proper checks
        return np.random.rand() > 0.95

    def successful_transition(self, observation):
        # Check for effective transitions between defense and attack
        # Dummy function, replace with proper checks
        return np.random.rand() > 0.9

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
