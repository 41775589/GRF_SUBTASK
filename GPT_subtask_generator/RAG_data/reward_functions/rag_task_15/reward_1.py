import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward based on ball passing precision and long-distance passing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_target_thresholds = np.linspace(0.3, 1.0, 10)  # Target distance thresholds
        self.pass_accuracy_reward = 0.05  # Reward for each threshold met without ball loss
        self._reset_pass_checkpoint()

    def _reset_pass_checkpoint(self):
        self.pass_checkpoint = np.zeros(10, dtype=bool)  # Track pass accuracy checkpoints

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._reset_pass_checkpoint()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'pass_checkpoint': self.pass_checkpoint,
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.pass_checkpoint = from_pickle['CheckpointRewardWrapper']['pass_checkpoint']
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_precision_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Calculate distance for long passes
            ball_dist = np.linalg.norm(o['ball'] - o['ball_direction'])

            for idx, threshold in enumerate(self.pass_target_thresholds):
                if ball_dist > threshold and not self.pass_checkpoint[idx] and o['ball_owned_team'] == 0:
                    # Reward for meeting each distance threshold without losing the ball
                    components["pass_precision_reward"][rew_index] += self.pass_accuracy_reward
                    self.pass_checkpoint[idx] = True

            reward[rew_index] += components["pass_precision_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
