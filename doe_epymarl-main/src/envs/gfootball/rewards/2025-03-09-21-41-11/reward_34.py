import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to add complex offensive strategies' rewards."""

    def __init__(self, env):
        super().__init__(env)
        self.passing_reward = 0.1
        self.dribbling_reward = 0.2
        self.shooting_reward = 0.3

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "shoot_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]
            sticky_actions = o['sticky_actions']
            mode = o['game_mode']

            # Reward for successful passes in open play
            if mode == 0 and sticky_actions[6]:  # Assuming 6 is pass in action set
                components['pass_reward'][i] = self.passing_reward
            
            # Reward for dribbling
            if mode == 0 and sticky_actions[9]:  # Assuming 9 is dribble in action set
                components['dribble_reward'][i] = self.dribbling_reward
                
            # Reward for shooting towards goal
            if mode == 0 and (sticky_actions[4] or sticky_actions[3]):  # Assuming 4, 3 are shoot actions
                components['shoot_reward'][i] = self.shooting_reward
                
            # Aggregate the rewards
            reward[i] += (components['pass_reward'][i] +
                          components['dribble_reward'][i] +
                          components['shoot_reward'][i])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
