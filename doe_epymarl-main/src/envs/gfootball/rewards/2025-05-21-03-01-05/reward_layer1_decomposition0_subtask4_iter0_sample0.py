import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to augment the reward based on dribbling and ball control under pressure in wide defensive areas."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking dribble and stop actions
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()[0]  # Assume single agent for simplicity
        components = {
            "base_score_reward": reward.copy(),   # Base game score reward
            "dribbling_reward": [0.0]             # Additional reward for maintaining possession under pressure
        }

        # Check if the player has possession of the ball and is on the defensive half
        if observation['ball_owned_team'] == 0 and observation['right_team_roles'][observation['active']] == 3:
            # Encourage dribbling by increasing reward based on dribbling in defensive regions
            if 'dribble' in observation['sticky_actions'] and observation['sticky_actions'][9] == 1:
                components['dribbling_reward'] = [1.0]  # Positive reward for dribbling in defensive half
                reward[0] += components['dribbling_reward'][0]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)  # Apply the customized reward function
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
