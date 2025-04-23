import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a clearance reward focusing on keeping the ball safe under pressure."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.clearance_checkpoints = 5
        self.clearance_zone_thresholds = np.linspace(-1, 0, num=self.clearance_checkpoints + 1)[1:]
        self.clearance_reward_coefficient = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            ball_position = o['ball'][0]  # X position of the ball
            if 'ball_owned_team' not in o or o['ball_owned_team'] != 0:
                continue

            if ball_position < 0:  # The ball is in the defensive half
                zone = np.digitize(ball_position, self.clearance_zone_thresholds, right=False)
                additional_reward = (self.clearance_checkpoints + 1 - zone) * self.clearance_reward_coefficient
                components["clearance_reward"][i] += additional_reward
                reward[i] += additional_reward

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
