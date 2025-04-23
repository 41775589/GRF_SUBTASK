import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on defensive actions."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.foul_penalty = -0.5  # Penalty for fouling an opponent
        self.tackle_reward = 0.2  # Reward for executing a tackle
        self.ball_intercept_reward = 0.3  # Reward for intercepting the ball
        self.non_foul_tackles = 0  # Counter for tackles without fouling

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.non_foul_tackles = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'non_foul_tackles': self.non_foul_tackles
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.non_foul_tackles = from_pickle['CheckpointRewardWrapper']['non_foul_tackles']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "foul_penalty": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index, obs in enumerate(observation):
            # Penalize for fouls
            if obs['game_mode'] in [3, 6]:  # FreeKick or Penalty
                components["foul_penalty"][rew_index] = self.foul_penalty
                reward[rew_index] += self.foul_penalty

            if obs['ball_owned_team'] == obs['active'] and obs['sticky_actions'][6] or obs['sticky_actions'][7]:
                # Standing or sliding tackle
                components["tackle_reward"][rew_index] = self.tackle_reward
                reward[rew_index] += self.tackle_reward
                self.non_foul_tackles += 1

            if obs['ball_owned_team'] == -1 and obs['sticky_actions'][8]:  # Ball intercept attempt
                components["ball_intercept_reward"][rew_index] = self.ball_intercept_reward
                reward[rew_index] += self.ball_intercept_reward

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
