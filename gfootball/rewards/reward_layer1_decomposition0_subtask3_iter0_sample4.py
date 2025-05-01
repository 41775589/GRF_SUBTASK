import gym
import numpy as np


class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for shot precision and effective dribbling in close-range scenarios."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._shot_precision_factor = 0.5  # Reward factor for precise shots
        self._effective_dribble_factor = 0.3  # Reward factor for effective dribbling
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_precision_reward": [0.0] * len(reward),
                      "effective_dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Add reward for shot precision
            if 'ball_direction' in o and 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                # Ball speed and angle should indicate a directed shot
                ball_speed = np.linalg.norm(o['ball_direction'][0:2])
                goal_dist = 1.0 - o['ball'][0]
                if ball_speed > 2.0 and goal_dist < 0.2:
                    components['shot_precision_reward'][rew_index] = self._shot_precision_factor
                    reward[rew_index] += components['shot_precision_reward'][rew_index]

            # Add reward for effective dribbling
            if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                dribble_action = o['sticky_actions'][9]
                if dribble_action == 1:  # Agent is dribbling
                    proximity_adv_opponents = np.any(np.abs(o['left_team'][:, 0] - o['ball'][0]) < 0.1)
                    if proximity_adv_opponents:
                        components['effective_dribble_reward'][rew_index] = self._effective_dribble_factor
                        reward[rew_index] += components['effective_dribble_reward'][rew_index]

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
