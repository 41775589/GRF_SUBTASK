import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to improve agents' attacking skills with checkpoints for creative offensive play and pressure adaptation."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_play_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            obs = observation[idx]
            # Reward for creative offensive play
            if obs['ball_owned_team'] == 0:  # Check if my team has the ball
                ball_x, ball_y = obs['ball'][0], obs['ball'][1]
                if ball_x > 0 and ball_y < 0.42 and ball_y > -0.42:  # Ball in opponent's half and not out of bounds
                    distance_to_goal = abs(1 - ball_x)  # Closer to goal on x-axis
                    components["offensive_play_reward"][idx] = (1 - distance_to_goal) * 0.1  # More reward closer to goal
                    reward[idx] += components["offensive_play_reward"][idx]

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
