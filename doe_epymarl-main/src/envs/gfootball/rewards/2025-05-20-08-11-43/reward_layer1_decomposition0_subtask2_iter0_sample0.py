import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for mastering intercepting and clearing the ball."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.intercept_reward = 1.0
        self.clear_ball_reward = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        if observation is None:
            return reward, {"base_score_reward": reward}

        components = {"base_score_reward": reward.copy(), 
                      "intercept_reward": 0.0,
                      "clear_ball_reward": 0.0}

        o = observation[0]
        
        # Check if our agent has just intercepted the ball.
        if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
            components["intercept_reward"] = self.intercept_reward
            reward += components["intercept_reward"]
        
        # Check for action 'Sliding' if the ball is cleared from a dangerous area
        if 'ball' in o and o['sticky_actions'][9]:  # Assuming index 9 is 'Sliding'
            if np.linalg.norm(o['ball'][:2]) < 0.3:  # Assuming dangerous area is near our goal
                components["clear_ball_reward"] = self.clear_ball_reward
                reward += components["clear_ball_reward"]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = reward
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
