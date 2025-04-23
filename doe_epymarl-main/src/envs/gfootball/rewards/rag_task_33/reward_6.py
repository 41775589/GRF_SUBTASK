import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effectively taking shots from outside the penalty box.
    This encourages players to learn long-range shooting, beating defenders, and optimal shooting decision-making.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.outside_penalty_box_reward = 0.05
        self.max_distance = 0.8  # approximately outside the penalty box
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['shoot_from_distance'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['shoot_from_distance']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "distance_shoot_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball'][0]  # the x-coordinate of the ball

            # Determine if the ball is being shot from outside the 'penalty box' boundary
            if abs(ball_position) > self.max_distance:
                components["distance_shoot_reward"][rew_index] = self.outside_penalty_box_reward
                reward[rew_index] += components["distance_shoot_reward"][rew_index]

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
