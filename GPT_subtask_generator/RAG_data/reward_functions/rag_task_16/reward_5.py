import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing high-precision high passes."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_thres = 0.3  # Estimating a threshold for high pass
        self.reward_for_pass = 0.5  # Reward for successful high pass
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward}

        components = {"base_score_reward": reward.copy()}
        
        for rew_index, o in enumerate(observation):
            # Check for high pass action, which depends on ball direction and high Y direction
            ball_direction = o['ball_direction']
            ball_pos = o['ball']
            high_pass_condition = ball_direction[1] > self.high_pass_thres and ball_pos[2] > 0.1
            
            if high_pass_condition:
                # Apply a reward when a high pass is detected
                reward[rew_index] += self.reward_for_pass
                components.setdefault("high_pass_reward", []).append(self.reward_for_pass)
            else:
                components.setdefault("high_pass_reward", []).append(0)
                
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
