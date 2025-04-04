import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering long passes."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_threshold = 0.3  # Threshold distance to consider a pass 'long'

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            obs = observation[i]
            if 'ball_owned_team' in obs and obs['ball_owned_team'] == 1:  # Right team has the ball
                if 'ball_direction' in obs:
                    ball_travel_distance = np.linalg.norm(obs['ball_direction'][:2])
                    if ball_travel_distance >= self.long_pass_threshold:
                        # Reward agents for making a long pass
                        components["long_pass_reward"][i] = 0.5
                        reward[i] += components["long_pass_reward"][i]

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
