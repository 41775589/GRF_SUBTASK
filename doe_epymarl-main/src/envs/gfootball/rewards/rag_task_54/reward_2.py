import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on enhancing the effectiveness of collaborative plays between shooters and passers to maximize scoring opportunities.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.passes_per_episode = 0
        self.shots_on_goal_per_episode = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_reward_coefficient = 0.05
        self.shot_reward_coefficient = 0.1

    def reset(self):
        self.passes_per_episode = 0
        self.shots_on_goal_per_episode = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['passes_per_episode'] = self.passes_per_episode
        state['shots_on_goal_per_episode'] = self.shots_on_goal_per_episode
        return state

    def set_state(self, state):
        self.env.set_state(state)
        self.passes_per_episode = state.get('passes_per_episode', 0)
        self.shots_on_goal_per_episode = state.get('shots_on_goal_per_episode', 0)
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, player_obs in enumerate(observation):
            # Check if a passing event occurred
            if player_obs['game_mode'] == 2 and player_obs['ball_owned_team'] == 1:  # Assuming '2' is a passing event
                self.passes_per_episode += 1
                components["passing_reward"][rew_index] = self.passing_reward_coefficient

            # Check if a shot on goal occurred
            if player_obs['game_mode'] == 6 and player_obs['ball_owned_team'] == 1:  # Assuming '6' is a shot on goal
                self.shots_on_goal_per_episode += 1
                components["shot_reward"][rew_index] = self.shot_reward_coefficient
            
            # Update reward
            reward[rew_index] += components["passing_reward"][rew_index] + components["shot_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
