import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for improving dribbling skills against the goalkeeper."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribbling_progress_reward = 0.05

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_progress_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o['game_mode'] != 0:  # Only consider normal play mode
                continue

            # Encourage dribbling actions near the opponent's goal
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                x_ball = o['ball'][0]
                y_ball = abs(o['ball'][1])  # Focus only on Y distance from centerline

                if x_ball > 0.5 and y_ball < 0.2:
                    # Closer to the goal and centralized in the field
                    distance_to_goal = 1 - x_ball  # 0 at goal line, 0.5 at mid
                    components["dribbling_progress_reward"][rew_index] = self.dribbling_progress_reward * (1 - distance_to_goal)
                    reward[rew_index] += components["dribbling_progress_reward"][rew_index]

            # Penalize losing the ball
            if o['ball_owned_team'] != 0 and self.sticky_actions_counter[9]:
                reward[rew_index] -= 0.05  # penalty for losing the ball while dribbling

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                self.sticky_actions_counter[i] = action

        return observation, reward, done, info
