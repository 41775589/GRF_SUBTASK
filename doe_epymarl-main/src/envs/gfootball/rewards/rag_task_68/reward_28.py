import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward focused on offensive skills: shooting, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.shooting_reward = 0.2
        self.dribbling_reward = 0.1
        self.passing_reward = 0.15
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'shooting_reward': self.shooting_reward,
            'dribbling_reward': self.dribbling_reward,
            'passing_reward': self.passing_reward
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['CheckpointRewardWrapper']
        self.shooting_reward = state_data['shooting_reward']
        self.dribbling_reward = state_data['dribbling_reward']
        self.passing_reward = state_data['passing_reward']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            player_obs = observation[rew_index]
            ball_ownership = player_obs['ball_owned_team'] == 1 and player_obs['ball_owned_player'] == player_obs['active']

            if ball_ownership:
                # Dribbling reward, active when moving with the ball
                if np.any(player_obs['sticky_actions'][8:]) == 1:  # sprint or dribble
                    reward[rew_index] += self.dribbling_reward
                    components["dribbling_reward"][rew_index] = self.dribbling_reward

                # Calculate distance to opponent's goal (simple approximation)
                goal_distance = 1 - player_obs['ball'][0]  # considering goal at x = 1
                # Shoot if close to the goal and rewarded for attempting to shoot.
                if goal_distance < 0.2:
                    reward[rew_index] += self.shooting_reward * (0.2 - goal_distance)
                    components["shooting_reward"][rew_index] = self.shooting_reward * (0.2 - goal_distance)

            # Passing, specifically looking for changes in ball ownership
            if player_obs['ball_owned_team'] == 1 and np.any(player_obs['sticky_actions'][6:8]) == 1:  # long or high passes
                reward[rew_index] += self.passing_reward
                components["passing_reward"][rew_index] = self.passing_reward
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
