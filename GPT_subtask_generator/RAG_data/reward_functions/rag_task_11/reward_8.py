import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for offensive capabilities with fast-paced maneuvers and precision finishing."""

    def __init__(self, env):
        super().__init__(env)  # Initialize the gym RewardWrapper with the environment.
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To count sticky actions for precise control measurement.
        self.goal_thresholds = np.linspace(0, 1, num=5)  # Checkpoints from midfield to opponent's goal.
        self.acquired_thresholds = [False] * 2  # To check if thresholds have been crossed.

    def reset(self):
        """Reset environment and counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.acquired_thresholds = [False] * 2
        return self.env.reset()

    def reward(self, reward):
        """Customized reward function to improve offensive skills."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "fast_paced_maneuver_reward": [0.0] * len(reward),
                      "precision_finishing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball']  # Position of the ball.
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            distance_to_goal = abs(player_pos[0] - 1.0)  # Simplified linear distance to opponent's goal on x-axis.

            # Reward for progressing towards the goal.
            for i, thresh in enumerate(self.goal_thresholds):
                if ball_pos[0] > thresh and not self.acquired_thresholds[i]:
                    components['fast_paced_maneuver_reward'][rew_index] += 0.2  # Incremental reward for each threshold.
                    self.acquired_thresholds[i] = True

            # Precision reward when in final third and controls the ball towards a shot.
            if ball_pos[0] > 0.66 and o['ball_owned_team'] == o['active']:
                components['precision_finishing_reward'][rew_index] = 1.0

            # Combine rewards
            total_reward = (reward[rew_index] +
                            components['fast_paced_maneuver_reward'][rew_index] +
                            components['precision_finishing_reward'][rew_index])

            reward[rew_index] = total_reward

        return reward, components

    def step(self, action):
        """Step through environment, capture data, and adjust reward based on new wrapper."""
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
        return obs, reward, done, info
