import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for attacking plays in creative offensive football scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._finishing_zones = 5
        self._checkpoint_rewards = [0.2, 0.4, 0.6, 0.8, 1.0]  # Gradually increase as closing to goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.total_reward = 0.0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_reward = np.array(reward)
        finishing_reward = np.zeros_like(base_reward)

        if observation is None:
            return base_reward, {"base_score_reward": base_reward.tolist()}

        for idx in range(len(base_reward)):
            player_obs = observation[idx]
            ball_pos = player_obs['ball']
            ball_owned_team = player_obs['ball_owned_team']
            ball_owned_player = player_obs['ball_owned_player']
            active_player = player_obs['active']  # assuming active is the active player index

            # Reward for controlling the ball and moving towards opponent's goal
            if ball_owned_team == 0 and active_player == ball_owned_player:
                x_position = ball_pos[0]
                zone_reward = self._calculate_finishing_reward(x_position)
                finishing_reward[idx] += zone_reward
        
            # Check for dribble or close control in advanced areas:
            if player_obs['sticky_actions'][9] == 1:  # Assuming the index 9 represents dribble
                finishing_reward[idx] += 0.05  # Small bonus for keeping the ball under pressure

        total_rewards = base_reward + finishing_reward
        self.total_reward += np.sum(total_rewards)
        return total_rewards, {
            "base_score_reward": base_reward.tolist(),
            "finishing_reward": finishing_reward.tolist(),
            "total_reward": self.total_reward
        }

    def _calculate_finishing_reward(self, x_position):
        """Calculate the reward based on x position on the pitch."""
        segment_length = 1.0 / self._finishing_zones
        for i, checkpoint in enumerate(self._checkpoint_rewards):
            if x_position > segment_length * i:
                reward = checkpoint
            else:
                break
        return reward

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, reward_components = self.reward(reward)
        info['total_reward'] = sum(reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(value) if isinstance(value, list) else value
        return observation, reward, done, info
