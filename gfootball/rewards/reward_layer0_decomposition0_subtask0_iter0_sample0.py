import gym
import numpy as np


class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focusing on attacking skills and pressure handling."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.base_reward_multiplier = 0.1
        self.possession_reward = 0.2
        self.goal_area_entry_reward = 1.0
        self.stressful_situation_multiplier = 0.5

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
        components = {
            "base_score_reward": reward.copy(),
            "possession_reward": [0.0] * len(reward),
            "goal_area_entry": [0.0] * len(reward),
            "stressful_situation_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for possession
            if o['ball_owned_team'] == 0:  # Assuming team 0 is the team we want to reward
                components["possession_reward"][rew_index] = self.possession_reward
                reward[rew_index] += components["possession_reward"][rew_index]

            # Extra reward when entering opponent's goal area
            ball_x, ball_y = o['ball'][0], o['ball'][1]
            if ball_x > 0.7 and abs(ball_y) < 0.2:  # Assuming right side is the opponent's side
                components["goal_area_entry"][rew_index] = self.goal_area_entry_reward
                reward[rew_index] += components["goal_area_entry"][rew_index]

            # Extra stress when close to opponent with the ball or near the match-end
            if (abs(ball_x - o['right_team'][:, 0]).min() < 0.1) or (o['steps_left'] < 200):
                components["stressful_situation_reward"][rew_index] = self.stressful_situation_multiplier
                reward[rew_index] += reward[rew_index] * components["stressful_situation_reward"][rew_index]

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
