import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for strategic positioning and possession changes."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_reward = 0.1
        self.defensive_position_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}  # Customize if there's state to save
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
        
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "passing_reward": [0.0] * len(reward),
            "defensive_position_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, obs in enumerate(observation):
            player_position = obs['right_team'][obs['active']]
            ball_position = obs['ball']

            if obs['game_mode'] in [2, 3]:  # FreeKick or GoalKick
                if np.linalg.norm(player_position - ball_position) < 0.2:  # Close to ball in strategic position
                    components["defensive_position_reward"][i] = self.defensive_position_reward
                    reward[i] += components["defensive_position_reward"][i]

            prev_ball_owner = obs.get('prev_ball_owned_player', -1)
            curr_ball_owner = obs['ball_owned_player']
            
            if prev_ball_owner != -1 and curr_ball_owner != -1 and prev_ball_owner != curr_ball_owner:
                if obs['ball_owned_team'] == obs['right_team_roles'][obs['active']]:  # If ball is owned by right team
                    components["passing_reward"][i] = self.passing_reward
                    reward[i] += components["passing_reward"][i]

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
