import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful long passes and accuracy in specific game situations."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_quality_rewards = np.zeros(10, dtype=float)  # Tracks the quality of passes
        self.pass_distances = np.linspace(0.2, 1.0, num=10)    # Define checkpoints for pass distances
        self.long_pass_weight = 0.5                             # Additional reward weight for successful long passes
        self.accuracy_weight = 0.5                              # Reward weight for accuracy
    
    def reset(self):
        self.pass_quality_rewards = np.zeros(10, dtype=float)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_pass_quality'] = self.pass_quality_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_quality_rewards = from_pickle['CheckpointRewardWrapper_pass_quality']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": np.zeros_like(reward),
                      "accuracy_reward": np.zeros_like(reward)}
        
        if observation is None:
            return reward, components

        for obs in observation:
            ball_pos = obs['ball'][:2]  # Extract x, y
            player_pos = obs['left_team'] if obs['ball_owned_team'] == 0 else obs['right_team']
            active_player_idx = obs['active']
            if active_player_idx != -1:
                active_player_pos = player_pos[active_player_idx]
                distance = np.linalg.norm(ball_pos - active_player_pos)
                
                # Check for long pass rewards
                for i, threshold in enumerate(self.pass_distances):
                    if distance >= threshold:
                        # Reward for distance covered by pass adjusted by weight
                        components['long_pass_reward'] += self.long_pass_weight * (1 / (i + 1))
                        self.pass_quality_rewards[i] += 1  # Increment quality count for passes matching this distance

                # Calculate accuracy component based on ball's landing position close to teammate
                if obs['ball_owned_player'] != active_player_idx and obs['ball_owned_team'] in [0, 1]:
                    components['accuracy_reward'] += self.accuracy_weight * (1 - distance)
        
        total_reward = reward + components['long_pass_reward'] + components['accuracy_reward']
        return total_reward.tolist(), components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info['component_' + key] = sum(value)
        return observation, reward, done, info
