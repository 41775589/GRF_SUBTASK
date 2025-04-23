import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds strategic play rewards focusing on maintaining ball control and exploiting open spaces."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for ball control and exploitation
        self.control_rewards = 0.05  # Reward increment for maintaining control under pressure
        self.exploit_open_space_reward = 0.1  # Reward for moving into open spaces strategically

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "control_under_pressure": [0.0] * len(reward),
                      "exploit_open_space": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            team = 'left_team' if o['ball_owned_team'] == 0 else 'right_team'
            player_pos = o[team][o['active']]
            nearest_opponent_dist = min([np.linalg.norm(player_pos - pos) for pos in o['right_team' if team == 'left_team' else 'left_team']])
            # Reward for maintaining ball control under pressure
            if o['ball_owned_team'] in [0, 1] and o['ball_owned_player'] == o['active']:
                if nearest_opponent_dist < 0.15: # assuming a threshold distance for "pressure"
                    components["control_under_pressure"][i] += self.control_rewards
                    reward[i] += components["control_under_pressure"][i]

            # Reward for moving towards and exploiting open spaces
            if player_pos[0] > 0 and team == 'left_team' or player_pos[0] < 0 and team == 'right_team':
                open_space_distance = abs(player_pos[0])  # Simplified measure of space exploitation
                components["exploit_open_space"][i] += self.exploit_open_space_reward * open_space_distance
                reward[i] += components["exploit_open_space"][i]
            
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
