import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on effective high passes from midfield. 
       It focuses on ball placement and timing to create direct scoring opportunities."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_count = 0
        self.pass_quality_threshold = 0.8  # Hypothetical threshold for what we consider a 'quality' high pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_count = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['high_pass_count'] = self.high_pass_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.high_pass_count = from_pickle['high_pass_count']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] != 0 or o['ball_owned_team'] not in [0, 1]:
                # Not in normal play or ball not owned
                continue

            if o['ball_owned_team'] == 0 and 0.2 < o['ball'][0] < 0.7:
                # Midfield range for the left team
                target_distance = 0.99 - abs(o['ball'][0])
                distance = np.linalg.norm(o['ball'][:2] - [1, 0])  # Distance to opponent's goal

                if distance < target_distance and o['ball_direction'][2] > 0:
                    # Check if the pass is high enough (positive Z direction)
                    quality = o['ball'][3]  # Using 'ball_direction' Z at index 3 hypothetically
                    if quality > self.pass_quality_threshold:
                        components['high_pass_reward'][rew_index] += 0.5
                        self.high_pass_count += 1

        reward = [r + reward_modifier for r, reward_modifier in zip(reward, components['high_pass_reward'])]
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
