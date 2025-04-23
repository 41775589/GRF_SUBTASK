import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive behavior and counterattacks in football."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.threshold_distance_to_goal = 0.5  # Threshold for proximity to own goal to increase defensive rewards.
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
                      "defensive_position_reward": [0.0] * len(reward),
                      "counterattack_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Defensive reward based on being close to own goal and intercepting the ball
            if o['ball_owned_team'] == 0 and np.linalg.norm(o['left_team'][o['active']] + [1, 0]) < self.threshold_distance_to_goal:
                components["defensive_position_reward"][rew_index] += 0.2

            # Bonus for initiating counterattack: possessing the ball in own half and moving forward
            if o['ball_owned_team'] == 0 and o['left_team'][o['active']][0] < 0 and o['left_team_direction'][o['active']][0] > 0:
                components["counterattack_bonus"][rew_index] += 0.5

            reward[rew_index] = (reward[rew_index] + 
                                components["defensive_position_reward"][rew_index] +
                                components["counterattack_bonus"][rew_index])

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
