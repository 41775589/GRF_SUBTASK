import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for defensive actions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions_count = 0
        self.interception_reward = 0.3
        self.defensive_positioning_reward = 0.1
    
    def reset(self):
        """ Reset the environment and the interception counters """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions_count = 0
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_interceptions'] = self.interceptions_count
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.interceptions_count = from_pickle['CheckpointRewardWrapper_interceptions']
        return from_pickle
    
    def reward(self, reward):
        """ Modify the reward based on defensive metrics """
        # This method is called with a list [score_gain_for_left_team, score_gain_for_right_team]
        # from the environment after each step.
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o and o['ball_owned_team'] != -1:
                opposing_team = 'right_team' if o['ball_owned_team'] == 0 else 'left_team'
                player_position = o[opposing_team][o['ball_owned_player']]
                own_goal_position = [-1, 0] if opposing_team == 'right_team' else [1, 0]
                
                distance_to_goal = np.linalg.norm(player_position - own_goal_position)
                if distance_to_goal < 0.3:
                    components["positioning_reward"][rew_index] = self.defensive_positioning_reward
                    reward[rew_index] += self.defensive_positioning_reward

            # Check if the ball has been intercepted
            if o['game_mode'] in [3, 4, 5, 6] and self.last_observation['ball_owned_team'] != o['ball_owned_team']:
                components["interception_reward"][rew_index] = self.interception_reward
                reward[rew_index] += self.interception_reward
                self.interceptions_count += 1
       
        self.last_observation = observation
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
                if action:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
