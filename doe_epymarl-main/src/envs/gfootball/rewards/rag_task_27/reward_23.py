import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies rewards based on the agent's ability to intercept the ball and maintain defensive positioning,
    encouraging defensive responsiveness in high-pressure scenarios.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)        
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interception_reward = 0.5
        self.defensive_positioning_reward = 0.2

    def reset(self):
        """
        Reset stick actions counters when the environment is reset.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save additional state information for pickle.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Load additional state information from pickle.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Custom reward function to enhance learning of defensive skills by adding interception and defensive positioning rewards.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward),
                      "defensive_positioning_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Interception Reward: ball ownership switched from opposing team to agent's team
            if o['ball_owned_team'] != self.prev_ball_owned_team and o['ball_owned_team'] == o['active']:
                if self.prev_ball_owned_team != 0:
                    components["interception_reward"][rew_index] = self.interception_reward
                    reward[rew_index] += components["interception_reward"][rew_index]

            # Defensive Positioning Reward: encourage positions closer to the own goal when opponent has the ball
            if o['ball_owned_team'] == 1 - o['active']:  # Ball is with the opponent
                player_pos = o['left_team'][o['active']] if o['active'] == 0 else o['right_team'][o['active']]
                goal_y = -0.42
                distance_to_goal = abs(player_pos[1] - goal_y)
                if distance_to_goal < 0.5:  # The closer to goal, the higher the reward
                    components["defensive_positioning_reward"][rew_index] = self.defensive_positioning_reward * (0.5 - distance_to_goal)
                    reward[rew_index] += components["defensive_positioning_reward"][rew_index]

            self.prev_ball_owned_team = o['ball_owned_team']

        return reward, components

    def step(self, action):
        """
        Perform a step in the environment, intercept the reward, and add additional information.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action

        return observation, reward, done, info
