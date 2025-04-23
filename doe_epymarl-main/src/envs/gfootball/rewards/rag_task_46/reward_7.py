import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """This wrapper enhances the reward for successful tackling and ball recovery without fouls during normal and set-piece plays."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_reward = 0.2
        self.tackle_penalty = -0.1
        self.recovered_ball_reward = 0.3
        self.possession_change_reward = 0.1

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
                      "tackle_success_reward": [0.0] * len(reward),
                      "recovered_ball_reward": [0.0] * len(reward),
                      "possession_change_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        ball_owned_team_start = self.prev_observation.get('ball_owned_team', -1) if self.prev_observation else -1

        for i, obs in enumerate(observation):
            components["base_score_reward"][i] = reward[i]
            game_mode = obs['game_mode']
            tackle_action = obs['sticky_actions'][9]  # Assume index 9 corresponds to tackling
            
            # Reward for successful tackles
            if tackle_action and obs['game_mode'] == 0:  # Normal play mode
                if obs['ball_owned_team'] == -1 and ball_owned_team_start != -1:  # Ball was owned and now it's free
                    reward[i] += self.tackle_success_reward
                    components["tackle_success_reward"][i] = self.tackle_success_reward
                elif obs['ball_owned_team'] != ball_owned_team_start:
                    reward[i] += self.possession_change_reward
                    components["possession_change_reward"][i] = self.possession_change_reward
            elif tackle_action and game_mode in [2, 3, 4, 5, 6]:  # Set-pieces
                if obs['ball_owned_team'] == -1 and ball_owned_team_start != -1:
                    reward[i] += self.tackle_success_reward
                    components["tackle_success_reward"][i] = self.tackle_success_reward
                elif obs['ball_owned_team'] != ball_owned_team_start:
                    reward[i] += self.possession_change_reward
                    components["possession_change_reward"][i] = self.possession_change_reward

            # Reward for recovering the ball without foul
            if obs['ball_owned_team'] == 0 and ball_owned_team_start == -1:  # Team 0 recovers the ball from -1 (free ball)
                reward[i] += self.recovered_ball_reward
                components["recovered_ball_reward"][i] = self.recovered_ball_reward

        self.prev_observation = observation
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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
