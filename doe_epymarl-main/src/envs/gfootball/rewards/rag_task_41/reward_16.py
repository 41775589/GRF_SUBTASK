import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides rewards geared toward enhancing attacking skills by promoting
    specialized training in finishing, creative play, and adaptive match-like responses.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goal_zones = np.linspace(-0.42, 0.42, 5)  # Divide y-axis near goal into zones
        self.zone_rewards = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # higher reward closer to goal
        self.previous_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', self.sticky_actions_counter)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_play_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                ball_position = o['ball']
                ball_y = ball_position[1]
                if self.previous_ball_position is not None:
                    moved_toward_goal = self.previous_ball_position[1] < ball_y <= 0.42
                    if moved_toward_goal:
                        # Calculate zone reward
                        zone_index = np.digitize(ball_y, self.goal_zones, right=True) - 1
                        zone_index = max(0, min(zone_index, len(self.goal_zones) - 1))
                        components['offensive_play_reward'][i] += self.zone_rewards[zone_index]
                self.previous_ball_position = ball_position
                reward[i] += components['offensive_play_reward'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for idx, act in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{idx}"] = act

        return observation, reward, done, info
