import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a shooting precision reward for close range scenarios near the goal."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
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
                      "precision_shooting_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Adjusted reward strategy for efforts at goal in close range
            ball_pos = o['ball']
            player_pos = o['right_team'][o['active']] if o['ball_owned_team'] == 1 else o['left_team'][o['active']]
            distance_to_goal = abs(ball_pos[0] - 1 if o['ball_owned_team'] == 0 else ball_pos[0] + 1)

            # Checking for close distance to goal and possession
            if o['ball_owned_team'] == o['active_team'] and distance_to_goal < 0.2:
                # Calculating reward based on angle and shooting power
                # assuming perfect shooting angle is directly at center of the goal (-0.044 to 0.044)
                goal_center_y = 0.0
                angle_error = abs(ball_pos[1] - goal_center_y)
                angle_penalty = max(0.1, min(angle_error / 0.044, 1.0))  # Normalize and limit
                components["precision_shooting_reward"][rew_index] = 1.0 - angle_penalty

                reward[rew_index] += components["precision_shooting_reward"][rew_index]

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
