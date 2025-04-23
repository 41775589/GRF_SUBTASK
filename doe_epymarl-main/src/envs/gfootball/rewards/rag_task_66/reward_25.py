import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides a specific reward function for mastering short passes
    under defensive pressure focusing on ball retention and effective distribution."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Specific game-related thresholds
        self.pass_completion_bonus = 0.2
        self.ball_possession_bonus = 0.1
        self.ball_lost_penalty = -0.1

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
        # Get recent observations from the environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_completion_bonus": [0.0] * len(reward),
                      "ball_possession_bonus": [0.0] * len(reward),
                      "ball_lost_penalty": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # init action flags
            has_ball = o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']
            just_passed = any(o['sticky_actions'][1:4])  # assuming indices for pass actions

            # Reward for maintaining possession under pressure
            if has_ball:
                components["ball_possession_bonus"][rew_index] += self.ball_possession_bonus
                reward[rew_index] += self.ball_possession_bonus
            
            # Reward for successful pass completion
            if has_ball and just_passed:
                components["pass_completion_bonus"][rew_index] += self.pass_completion_bonus
                reward[rew_index] += self.pass_completion_bonus
            
            # Penalty if ball lost immediately after pass or receiving
            if not has_ball and just_passed:
                components["ball_lost_penalty"][rew_index] += self.ball_lost_penalty
                reward[rew_index] += self.ball_lost_penalty

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
