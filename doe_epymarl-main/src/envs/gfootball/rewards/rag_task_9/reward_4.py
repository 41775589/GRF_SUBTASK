import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A gym wrapper that adds rewards for successful offensive skills such as passing, shooting, and dribbling"""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Count of sticky actions
        
    def reset(self):
        """Reset the environment and the sticky action counter"""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """State getter with wrapper information"""
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """State setter with adjusting the sticky action counter"""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Modify reward based on offensive actions."""
        observation = self.env.unwrapped.observation()  # Get current observations
        components = {
            "base_score_reward": reward.copy(),
            "passing": [0.0] * len(reward),
            "shooting": [0.0] * len(reward),
            "dribbling": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            player_obs = observation[i]
            sticky_actions = player_obs['sticky_actions']

            # Reward for successful passes (short and long)
            if sticky_actions[0] or sticky_actions[1]:  # Assuming indices 0 and 1 are short and long pass actions
                components["passing"][i] = 0.05
                # Update sticky actions counter
                self.sticky_actions_counter[0] += sticky_actions[0]
                self.sticky_actions_counter[1] += sticky_actions[1]

            # Reward for successful shots
            if sticky_actions[2]:  # Assuming index 2 is the shooting action
                components["shooting"][i] = 0.1
                self.sticky_actions_counter[2] += sticky_actions[2]

            # Reward for successful dribbling
            if sticky_actions[9]:  # Assuming index 9 is the dribbling action
                components["dribbling"][i] = 0.03
                self.sticky_actions_counter[9] += sticky_actions[9]

            # Calculate the total reward
            reward[i] += components["passing"][i] + components["shooting"][i] + components["dribbling"][i]

        return reward, components

    def step(self, action):
        """Execute environment step and modify the reward"""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Add final and component rewards to info
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
