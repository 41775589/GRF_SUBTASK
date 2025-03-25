import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward function to better suit a hybrid midfielder/defender role,
    focusing on high passes, long passes, dribbling under pressure, and effective use of sprint.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, from_pickle):
        state = self.env.set_state(from_pickle)
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'possession_reward': [0.0] * len(reward),
                      'pass_reward': [0.0] * len(reward),
                      'sprint_reward': [0.0] * len(reward)}

        for idx in range(len(reward)):
            obs = observation[idx]
            
            # Reward for maintaining possession under pressure
            if ('ball_owned_team' in obs and obs['ball_owned_team'] == 0 and
                obs['sticky_actions'][9] == 1):  # Dribble
                components['possession_reward'][idx] += 0.02

            # Reward for successful high or long passes
            if obs['game_mode'] in [3, 4]:  # Game mode indicates a set piece or potential long pass scenario
                components['pass_reward'][idx] += 0.05

            # Reward for effective sprint usage transitions
            if obs['sticky_actions'][8] == 1:  # Sprint action is active
                components['sprint_reward'][idx] += 0.01

            # Integrate the additional rewards into the central reward
            reward[idx] = reward[idx] + sum([components[key][idx] for key in components])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
