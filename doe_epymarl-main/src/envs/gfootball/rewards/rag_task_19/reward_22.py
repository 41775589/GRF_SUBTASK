import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that emphasizes on defensive and midfield control, promoting strategic
    interplay between these areas. Rewards positive defensive actions and effective
    midfield management.
    """

    def __init__(self, env):
        super().__init__(env)
        self.defensive_actions = np.zeros(10, dtype=int)
        self.midfield_control = np.zeros(10, dtype=int)
        self.defensive_rewards = 0.05
        self.midfield_rewards = 0.03

    def reset(self):
        self.defensive_actions = np.zeros(10, dtype=int)
        self.midfield_control = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_actions'] = self.defensive_actions
        to_pickle['midfield_control'] = self.midfield_control
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_actions = from_pickle['defensive_actions']
        self.midfield_control = from_pickle['midfield_control']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "defensive_reward": [0.0] * len(reward),
                      "midfield_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for i, o in enumerate(observation):
            # Encourage defensive actions, particularly intercepting the ball in the defensive third
            if o['ball'][0] < -0.5 and o['ball_owned_team'] == 0 and o['active'] in o['left_team']: 
                components["defensive_reward"][i] += self.defensive_rewards

            # Encourage maintaining control in the midfield
            if -0.33 < o['ball'][0] < 0.33 and o['ball_owned_team'] == 0:
                components["midfield_reward"][i] += self.midfield_rewards

            # Update total reward with the additional components
            reward[i] += components["defensive_reward"][i]
            reward[i] += components["midfield_reward"][i]

        return reward, components

    def step(self, action):
        o, r, d, i = self.env.step(action)
        r, components = self.reward(r)
        i["final_reward"] = sum(r)
        for key, value in components.items():
            i[f"component_{key}"] = sum(value)
        
        return o, r, d, i
