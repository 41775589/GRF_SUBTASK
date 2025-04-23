import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards for offensive strategies including shooting, dribbling, and passing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward magnitudes
        self.shot_reward = 0.3
        self.dribble_reward = 0.1
        self.pass_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'StickyActionsCounter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['StickyActionsCounter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "shot_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward)
        }

        for rew_index, obs in enumerate(observation):
            if (obs['sticky_actions'][9] == 1):  # dribbling
                reward[rew_index] += self.dribble_reward
                components['dribble_reward'][rew_index] += self.dribble_reward

            # Reward shooting based on proximity to the goal (ball position x near 1 or -1)
            if (obs['ball_owned_team'] == 0 and obs['ball'][0] > 0.8) or (obs['ball_owned_team'] == 1 and obs['ball'][0] < -0.8):
                if (obs['sticky_actions'][10] == 1):  # shooting
                    reward[rew_index] += self.shot_reward
                    components['shot_reward'][rew_index] += self.shot_reward

            # Reward passing based on changes in ball ownership
            if (obs['game_mode'] in (5, 6)):  # assuming mode 5 and 6 are related to passing or ball transitions
                reward[rew_index] += self.pass_reward
                components['pass_reward'][rew_index] += self.pass_reward
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, component_values in components.items():
            info[f"component_{key}"] = sum(component_values)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += active

        return observation, reward, done, info
