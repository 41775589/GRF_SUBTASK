import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes offensive strategies including accurate shooting, 
       dribbling past opponents, and performing long and high passes."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define coefficients for additional rewards
        self.shooting_coefficient = 0.3
        self.dribbling_coefficient = 0.2
        self.passing_coefficient = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Reset sticky actions counter
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {"base_score_reward": reward.copy()}

        # Validate observations length matches the rewards length
        assert len(reward) == len(observation)
        
        components = {
            "base_score_reward": reward.copy(),
            "shooting_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward)
        }

        for i, ob in enumerate(observation):
            # Assess shooting skills (assuming shooting when close to goal with ball possession)
            if ob['ball_owned_team'] == ob['active'] and np.linalg.norm(ob['ball'][:2] - [1, 0]) < 0.1:
                components["shooting_reward"][i] = self.shooting_coefficient
            
            # Assess dribbling (using sticky action dribble and player with possession)
            if ob['ball_owned_team'] == ob['active'] and ob['sticky_actions'][9] == 1:
                components["dribbling_reward"][i] = self.dribbling_coefficient

            # Assess passing skills, high and long passes (assuming these when changing ball direction significantly)
            if ob['ball_owned_team'] == ob['active'] and np.linalg.norm(ob['ball_direction'][:2]) > 0.05:
                components["passing_reward"][i] = self.passing_coefficient

            # Aggregate the rewards with the components
            reward[i] += (components["shooting_reward"][i] +
                          components["dribbling_reward"][i] +
                          components["passing_reward"][i])
        
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
