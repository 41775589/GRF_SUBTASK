import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for maintaining strategic positions,
    ensuring effective transitions between defense and attack using all directional movements."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positional_rewards": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx, rew in enumerate(reward):
            o = observation[idx]
            # Define thresholds for strategic positioning rewards
            # Adjust coefficients as needed to balance the reward
            strategic_position_coefficient = 0.05
            pivot_action_coefficient = 0.02

            # Reward for strategic positioning (distance to the ball and opponents)
            ball_dist = np.linalg.norm(np.array(o['ball'][:2]) - np.array(o['right_team'][o['active']]))
            opponent_average_dist = np.mean([np.linalg.norm(np.array(o['right_team'][o['active']]) - opp_pos)
                                             for opp_pos in o['left_team']])
            components['positional_rewards'][idx] = strategic_position_coefficient / (ball_dist + 1e-5) + \
                                                    pivot_action_coefficient / (opponent_average_dist + 1e-5)

            # Aggregate the reward
            reward[idx] += components['positional_rewards'][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Append each component to the info dictionary for tracking
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions analysis
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
