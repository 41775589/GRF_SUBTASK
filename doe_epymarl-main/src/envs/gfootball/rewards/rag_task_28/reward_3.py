import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for dribbling skills improvement in 1v1 situations
    against the goalkeeper, with emphasis on feints and direction changes while maintaining ball control.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """
        Custom reward function to foster dribbling skills:
        1. Positive reward for successful dribble moves (detected by ball control maintenance and direction changes).
        2. Negative reward slightly for losing the ball control.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribble_reward": [0.0, 0.0]
        }
        for i, o in enumerate(observation):
            # Check if the active player is in control of the ball and is near the goalkeeper
            if o['ball_owned_team'] == 0 and abs(o['ball'][0]) > 0.7 and o['ball'][0] > 0:
                if o['active'] == o['ball_owned_player']:
                    # Reward based on maintaining ball control and executing dribble (sticky action 9)
                    if o['sticky_actions'][9] == 1:
                        components['dribble_reward'][i] = 0.05
                    # Increment reward if changing directions
                    if np.any(o['left_team_direction'][o['active'], :] != 0):
                        components['dribble_reward'][i] += 0.05
                else:
                    # Negative reward for losing the ball
                    components['dribble_reward'][i] -= 0.01
            # Aggregate final reward calculation
            reward[i] = components['base_score_reward'][i] + components['dribble_reward'][i]

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
