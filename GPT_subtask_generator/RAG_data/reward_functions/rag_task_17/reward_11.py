import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering wide midfield responsibilities."""
    
    def __init__(self, env):
        super().__init__(env)
        # Initialize counters for sticking actions and high passes
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # considering 10 potential sticky actions
        self.high_passes_counter = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_passes_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        state['high_passes_counter'] = self.high_passes_counter
        return state

    def set_state(self, state):
        state_from_env = self.env.set_state(state)
        self.sticky_actions_counter = state_from_env.get('sticky_actions_counter', np.zeros(10, dtype=int))
        self.high_passes_counter = state_from_env.get('high_passes_counter', 0)
        return state_from_env

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
        
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward),
                      "high_pass_reward": [0.0] * len(reward)}

        for idx, o in enumerate(observation):
            # Check if the ball is near the wide areas of the field
            if abs(o['left_team'][self.env._agent_index][1]) > 0.3:  # more than 30% to the side of the field
                components['positioning_reward'][idx] = 0.1  # rewarding stepping into wide areas
                
            # Reward high passes
            if o['sticky_actions'][9] == 1:  # assuming index 9 corresponds to high passes action
                self.high_passes_counter += 1
                components['high_pass_reward'][idx] = 0.5 * self.high_passes_counter
                
            # Aggregate rewards
            reward[idx] += components['positioning_reward'][idx] + components['high_pass_reward'][idx]

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
