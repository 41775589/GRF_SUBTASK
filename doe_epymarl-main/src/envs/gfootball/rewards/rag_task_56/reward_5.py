import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A custom reward wrapper designed to enhance defensive play, with emphasis on goalkeeping and defending skills.
    The reward increases with successful ball interceptions, good positioning, and maintaining possession under pressure.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_reward = 0.5
        self.defender_reward = 0.3
        self.intercept_reward = 0.4
        self.possession_reward = 0.2

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
        # Obtain the current observations
        observation = self.env.unwrapped.observation()
        
        components = {
            "base_score_reward": reward.copy(),
            "goalkeeper_reward": [0.0] * len(reward),
            "defender_reward": [0.0] * len(reward),
            "intercept_reward": [0.0] * len(reward),
            "possession_reward": [0.0] * len(reward)
        }
        
        # This code block assumes that self.env.unwrapped.observation() returns structured observations including
        # player roles, ball ownership, etc., aligned with the exposed observations.
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Enhance rewards for goalkeeper actions
            if o['active'] == 0 and o['left_team_roles'][o['active']] == 0:  # Assuming 0 is the goalkeeper role index
                components["goalkeeper_reward"][rew_index] = self.goalkeeper_reward
        
            # Rewards for successful interceptions by defenders
            if o['ball_owned_team'] == 0 and o['left_team_active'][o['active']]:
                role = o['left_team_roles'][o['active']]
                if role in [1, 2, 3, 4]:  # Assuming these are defender roles
                    components["defender_reward"][rew_index] = self.defender_reward

            # Additional rewards for intercepting the ball
            if ('ball_owned_team' in o and 
                o['ball_owned_team'] == 0 and 
                o['ball_owned_player'] == o['active']):
                components["intercept_reward"][rew_index] = self.intercept_reward

            # Reward maintaining possession under pressure
            if o['ball_owned_team'] == 0 and o['left_team_active'][o['active']]:
                components["possession_reward"][rew_index] = self.possession_reward

            # Integrating all components into the final reward for this player
            reward[rew_index] += (components["goalkeeper_reward"][rew_index] +
                                  components["defender_reward"][rew_index] +
                                  components["intercept_reward"][rew_index] +
                                  components["possession_reward"][rew_index])

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
