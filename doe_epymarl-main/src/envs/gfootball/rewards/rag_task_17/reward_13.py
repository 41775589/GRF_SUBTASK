import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for high passes and intelligent positioning by midfield players 
    to stretch the opposition defense and create space.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Configuration parameters for the reward function
        self.high_pass_reward = 0.2  # Reward for making a high pass
        self.positioning_reward = 0.5  # Reward for good positioning
        self.positioning_threshold = 0.7  # Minimum y-coordinate for midfielders to be considered well-positioned
        
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
                      "high_pass_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_role = o['left_team_roles'][o['active']] if o['ball_owned_team'] == 0 else o['right_team_roles'][o['active']]
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            high_pass = o['sticky_actions'][8]  # Index 8 corresponds to 'action_sprint'
            y_pos = player_pos[1]  # Extracting only y-coordinate of the position

            # Reward for High Passes
            if high_pass and (player_role in {6, 7}):  # Assuming the roles 6 and 7 correspond to midfielders
                components["high_pass_reward"][rew_index] = self.high_pass_reward
                reward[rew_index] += components["high_pass_reward"][rew_index]
                
            # Reward for positioning for midfield roles
            if (player_role in {6, 7}) and o['ball_owned_team'] == 0 and y_pos >= self.positioning_threshold:
                components["positioning_reward"][rew_index] = self.positioning_reward
                reward[rew_index] += components["positioning_reward"][rew_index]

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
