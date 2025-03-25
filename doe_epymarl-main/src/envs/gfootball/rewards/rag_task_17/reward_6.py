import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on midfielders' high passes and effective lateral positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passes_completed = 0
        self.positioning_quality = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passes_completed = 0
        self.positioning_quality = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'passes_completed': self.passes_completed,
            'positioning_quality': self.positioning_quality
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        retrieved = from_pickle['CheckpointRewardWrapper']
        self.passes_completed = retrieved['passes_completed']
        self.positioning_quality = retrieved['positioning_quality']
        return from_pickle

    def reward(self, reward):
        original_reward = reward.copy()
        observation = self.env.unwrapped.observation()

        component_rewards = {
            "base_score_reward": original_reward,
            "high_pass_reward": [0.0] * len(reward),
            "lateral_positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, component_rewards

        for i in range(len(reward)):
            obs = observation[i]
            if obs['game_mode'] == 0 and obs['ball_owned_team'] == obs['active']:
                # Detecting high passes
                if obs['sticky_actions'][9] == 1:  # Action for high pass
                    if obs['ball_direction'][1] > 0.1 or obs['ball_direction'][1] < -0.1:  # Lateral high pass
                        component_rewards['high_pass_reward'][i] = 0.2
                        reward[i] += component_rewards['high_pass_reward'][i]
                        self.passes_completed += 1
                # Enhancing lateral movement to stretch the defense
                if abs(obs['left_team'][obs['active']][1]) > 0.3:  # Near sidelines
                    component_rewards['lateral_positioning_reward'][i] = 0.1
                    reward[i] += component_rewards['lateral_positioning_reward'][i]
                    self.positioning_quality += 1

        return reward, component_rewards

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
