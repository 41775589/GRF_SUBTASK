import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on offensive strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.checkpoint_counter = {}
        self.total_checkpoints = 10
        self.shooting_reward = 0.5
        self.dribbling_reward = 0.3
        self.passing_reward = 0.2

    def reset(self):
        self.checkpoint_counter = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.checkpoint_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.checkpoint_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            obs = observation[i]
            
            # Encounter a goal which is a success for offensive actions
            if obs['score'][1] > obs['score'][0]:  # Assuming agent team is team 1 (right team)
                if self.checkpoint_counter.get(i, 0) < self.total_checkpoints:
                    reward[i] += self.shooting_reward
                    components["shooting_reward"][i] = self.shooting_reward
                self.checkpoint_counter[i] = self.total_checkpoints
            
            # Add rewards for dribbling if the player is controlling the ball and evading others
            if obs['ball_owned_team'] == 1 and obs['active'] == obs['ball_owned_player']:
                reward[i] += self.dribbling_reward
                components["dribbling_reward"][i] = self.dribbling_reward

            # Add rewards for successful passing in advanced field positions
            if obs['ball_owned_team'] == 1 and np.any(obs['sticky_actions'][[1, 3, 5, 7]]):
                distance_to_goal = np.abs(obs['ball'][0] - 1)
                if distance_to_goal < 0.5:  # closer to opponent's goal
                    reward[i] += self.passing_reward
                    components["passing_reward"][i] = self.passing_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
