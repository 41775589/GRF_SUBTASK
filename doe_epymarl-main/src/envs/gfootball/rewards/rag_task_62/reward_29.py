import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a shooting angle and timing optimization reward."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._score_increase = 1.0
        self._near_goal_bonus = 0.2
        self._shooting_position_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "near_goal_bonus": [0.0] * len(reward),
                      "shooting_position_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Check if near the opponent's goal and in control of the ball
            if o['ball_owned_team'] == o['active'] and np.abs(o['ball'][0]) > 0.8:
                components["near_goal_bonus"][rew_index] = self._near_goal_bonus
                reward[rew_index] += self._near_goal_bonus

            # Reward for proper shooting positions and decision timing
            if o['ball_owned_team'] == 1 and o['game_mode'] == 0:
                x_distance_to_goal = 1 - o['ball'][0]
                y_distance_to_goal = np.abs(o['ball'][1])

                # Encourages shooting straight towards the goal when close
                if x_distance_to_goal < 0.2 and y_distance_to_goal < 0.1:
                    shooting_score = self._shooting_position_reward / (x_distance_to_goal + 0.1)
                    components["shooting_position_reward"][rew_index] = shooting_score
                    reward[rew_index] += shooting_score

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
