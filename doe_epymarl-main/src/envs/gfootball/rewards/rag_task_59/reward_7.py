import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on goalkeeper coordination during high-pressure scenarios
    and efficient ball clearing to specific outfield players.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.clearing_rewards = 0.2  # Reward for clearing the ball efficiently
        self.backup_rewards = 0.3    # Reward for backing up the goalkeeper
        self.goalkeeper_position = -1  # Initialize goalkeeper position
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # sticky actions count

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_position = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'goalkeeper_position': self.goalkeeper_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.goalkeeper_position = from_pickle['CheckpointRewardWrapper']['goalkeeper_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearing_reward": [0.0] * len(reward),
                      "backup_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            goalie_role = 0  # Assuming goalie's role index is 0

            # Detect if the current active player is the goalkeeper
            if 'ball_owned_team' in o:
                team = 'left_team' if o['ball_owned_team'] == 0 else 'right_team'
                for p_idx, role in enumerate(o[team + '_roles']):
                    if role == goalie_role:
                        self.goalkeeper_position = p_idx
                        break
            
            # High-pressure scenario: ball is very close to the goalkeeper 
            ball_pos = o['ball'][:2]  # ignore z-coordinate
            goalie_pos = o[team][self.goalkeeper_position]
            dist_to_goalie = np.linalg.norm(ball_pos - goalie_pos)

            # Back-up the goalkeeper and clearing the ball
            if dist_to_goalie <= 0.1:  # define threshold for "high-pressure"
                if o['ball_owned_team'] == 0:  # if our team owns the ball
                    components["backup_reward"][rew_index] = self.backup_rewards
                    reward[rew_index] += components["backup_reward"][rew_index]
                else:  # clear the ball by goalkeeper
                    components["clearing_reward"][rew_index] = self.clearing_rewards
                    reward[rew_index] += components["clearing_reward"][rew_index]

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
