import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on offensive football skills."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_reward = 0.1
        self.shooting_reward = 0.2
        self.dribbling_reward = 0.1
        self.sprint_bonus = 0.05
        self.ball_control_bonus = 0.15

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_skill_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            skill_reward = 0

            # Evaluate passing
            if o['sticky_actions'][0] or o['sticky_actions'][1]:  # Short pass or long pass
                skill_reward += self.passing_reward

            # Evaluate shooting
            if o['sticky_actions'][2]:  # Shot
                skill_reward += self.shooting_reward

            # Evaluate dribbling and sprint
            if o['sticky_actions'][9]:  # Dribble
                skill_reward += self.dribbling_reward
            if o['sticky_actions'][8]:  # Sprint
                skill_reward += self.sprint_bonus

            # Bonus for ball control
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:  # The controlled player has the ball
                skill_reward += self.ball_control_bonus

            components["offensive_skill_reward"][rew_index] = skill_reward
            reward[rew_index] += skill_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
