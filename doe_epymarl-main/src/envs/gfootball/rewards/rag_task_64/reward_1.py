import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for successful high passes and crosses, encouraging dynamic attacking plays and improving spatial creation
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.5  # Reward for successful high passes
        self.cross_reward = 1.0  # Reward for successful crosses into the box

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
        observation = self.env.unwrapped.observation()  # Returned as a list of observations for each agent
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward),
            "cross_reward": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]  # Observation for the current agent
            ball_pos = o['ball']
            ball_direction = o['ball_direction']

            # Check if a high pass or cross has been achieved
            if ball_direction[2] > 0.1:  # Assumption that Z-direction velocity indicates a high ball
                if np.linalg.norm(ball_direction[:2]) > 0.5:  # Assumption for a strong pass
                    # Determine if it's a cross into the box
                    if abs(ball_pos[0]) > 0.7 and abs(ball_pos[1]) < 0.2:  # Ball in 'crossing' range near the goal box
                        components['cross_reward'][rew_index] = self.cross_reward
                    else:
                        components['pass_reward'][rew_index] = self.pass_reward

            # Assign reward based on the actions outcomes
            reward[rew_index] += components['pass_reward'][rew_index] + components['cross_reward'][rew_index]

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
            for i, act in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += act
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
