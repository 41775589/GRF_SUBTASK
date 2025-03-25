import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds offensive maneuver focused rewards based on quick attacks and dynamic adaption during various game phases."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.score_differential_reward = 0.1
        self.game_state_change_reward = 0.05
        self.ball_possession_change_reward = 0.2
    
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
                      "score_differential_reward": [0.0] * len(reward),
                      "game_state_change_reward": [0.0] * len(reward),
                      "ball_possession_change_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward Change if there is a score change
            components["score_differential_reward"][rew_index] = (o['score'][0] - o['score'][1]) * self.score_differential_reward
            reward[rew_index] += components["score_differential_reward"][rew_index]

            # Reward Change when game state changes, encouraging adaptation to different phases
            if self.env.previous_game_mode != o['game_mode']:
                components["game_state_change_reward"][rew_index] = self.game_state_change_reward
                reward[rew_index] += components["game_state_change_reward"][rew_index]
                self.env.previous_game_mode = o['game_mode']

            # Reward players for regaining possession of the ball
            if self.env.prev_ball_owned_team != o['ball_owned_team'] and o['ball_owned_team'] == o['left_team']:
                components["ball_possession_change_reward"][rew_index] = self.ball_possession_change_reward
                reward[rew_index] += components["ball_possession_change_reward"][rew_index]
            
            self.env.prev_ball_owned_team = o['ball_owned_team']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
