import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering the 'Sliding' tackle to dispossess opponents effectively."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward increment when the sliding tackle is successfully executed.
        self.sliding_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Modify the rewards based on successful use of defensive 'Sliding' tackle."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "sliding_tackle_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for idx, obs in enumerate(observation):
            # Checking if the ball is owned by opponent and current action is 'Sliding'
            if obs['ball_owned_team'] == 1 and obs['sticky_actions'][9] == 1:  # Assuming index 9 refers to 'Sliding'
                # Calculate the distance from the ball
                ball_pos = obs['ball'][:2]
                player_pos = obs['right_team'][obs['active']][:2] if obs['ball_owned_team'] == 1 else obs['left_team'][obs['active']][:2]
                distance = np.linalg.norm(ball_pos - player_pos)

                # Reward if distance is small enough, suggesting effective sliding
                if distance < 0.1:  # Threshold distance to consider effective tackling
                    components['sliding_tackle_reward'][idx] = self.sliding_reward

        total_rewards = [base + sliding for base, sliding in zip(components['base_score_reward'], components['sliding_tackle_reward'])]
        return total_rewards, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
