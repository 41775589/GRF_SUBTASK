import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward focused on defensive skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components
        
        intercept_bonus = 0.1  # Bonus reward for successful interceptions

        for agent_idx, agent_obs in enumerate(observation):
            # Enhance reward based on defensive actions especially interceptions
            # Check game mode to identify potential interception scenarios
            is_defensive_pressuring = (agent_obs['game_mode'] == 3 or agent_obs['game_mode'] == 5)

            # Active player's details
            active_player = agent_obs['active']
            player_pos = agent_obs['left_team'][active_player]
            ball_pos = agent_obs['ball']
            
            # Calculation of distance from ball to approx interception potential
            distance_from_ball = np.linalg.norm(ball_pos[:2] - player_pos)
            is_close_to_ball = distance_from_ball < 0.05

            # Reward more for defensive pressuring and being close to the ball
            if agent_obs['ball_owned_team'] != 0 and is_close_to_ball and is_defensive_pressuring:
                components["interception_reward"] = intercept_bonus
                reward[agent_idx] += intercept_bonus
            else:
                components["interception_reward"] = 0.0

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
