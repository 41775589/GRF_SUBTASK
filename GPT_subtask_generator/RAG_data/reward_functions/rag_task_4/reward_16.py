import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for dribbling skills, sprint, and evasion."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.last_player_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.last_player_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            ball_pos = o['ball'][:2]
            player_pos = o['left_team'][o['active']]

            # Sprint & ball control enhances dribbling in offensive positions
            sprint_active = o['sticky_actions'][8]
            dribble_active = o['sticky_actions'][9]

            if sprint_active:
                # Reward for using sprint effectively to evade and approach
                distance_traveled = np.linalg.norm(np.array(player_pos) - np.array(self.last_player_position))
                sprint_reward = distance_traveled * 0.02 if dribble_active else 0.01
                components.setdefault('sprint_reward', []).append(sprint_reward)
                reward[rew_index] += sprint_reward

            if dribble_active:
                # Reward for maintaining ball possession while dribbling forward
                dribble_reward = 0.05 if ball_pos[0] > self.last_ball_position[0] else -0.01
                components.setdefault('dribble_reward', []).append(dribble_reward)
                reward[rew_index] += dribble_reward

            # Updating for next step
            self.last_ball_position = ball_pos
            self.last_player_position = player_pos

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
