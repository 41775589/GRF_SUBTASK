import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a strategic offensive reward based on shooting, dribbling, and passing."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.shooting_reward = 1.0  # Reward for shots taken close to the goal
        self.dribbling_reward = 0.5  # Reward for dribbling near opponents
        self.passing_reward = 0.2  # Reward for successful long and high passes

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        for idx in range(len(reward)):
            o = observation[idx]
            if 'ball_owned_team' in o and o['ball_owned_team'] == o['active']:
                ball_pos = o['ball']
                # Generate shooting reward if near the opponent's goal
                if ball_pos[0] > 0.5:  # Assuming 1 is the opponent's goal x-position
                    components['shooting_reward'][idx] = self.shooting_reward
                    reward[idx] += components['shooting_reward'][idx]

                # Generate dribbling reward if close to any opponent and has the ball
                if np.any(np.linalg.norm(o['right_team'] - o['ball'], axis=1) < 0.1):
                    components['dribbling_reward'][idx] = self.dribbling_reward
                    reward[idx] += components['dribbling_reward'][idx]

                # Passing rewards for long and high passes
                if 'ball_direction' in o and np.linalg.norm(o['ball_direction'][:2]) > 0.3:
                    components['passing_reward'][idx] = self.passing_reward
                    reward[idx] += components['passing_reward'][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
