import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defensive coordination near the penalty area."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_coordination_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            # Calculate reward based on player positions when the ball is near their goal
            if obs['game_mode'] in {3, 4}:
                # Define a region near the penalty area
                if obs['ball'][0] < -0.8:
                    ball_owners_team = obs['ball_owned_team']
                    if ball_owners_team == 0:  # Defensive scenario for left team
                        players_near_ball = np.linalg.norm(obs['left_team'] - obs['ball'], axis=1) < 0.1
                        n_defenders_close = np.sum(players_near_ball)
                        if n_defenders_close > 1:
                            components["defensive_coordination_reward"][rew_index] += 0.05 * n_defenders_close
                    elif ball_owners_team == 1:  # Defensive scenario for the right team
                        players_near_ball = np.linalg.norm(obs['right_team'] - obs['ball'], axis=1) < 0.1
                        n_defenders_close = np.sum(players_near_ball)
                        if n_defenders_close > 1:
                            components["defensive_coordination_reward"][rew_index] += 0.05 * n_defenders_close

            # Total reward for this agent
            reward[rew_index] += components["defensive_coordination_reward"][rew_index]

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
