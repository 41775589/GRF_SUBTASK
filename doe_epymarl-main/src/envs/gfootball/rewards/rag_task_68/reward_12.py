import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that optimizes offensive gameplay by rewarding accurate shooting,
    dribbling to evade opponents, and effective passing to break defensive lines."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_reward = 0.3
        self.dribbling_reward = 0.2
        self.passing_reward = 0.1

    def reset(self):
        """Reset the environment and the sticky_action_counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of the env and additional wrappers."""
        to_pickle['CheckpointRewardWrapper'] = None  # Add specific elements if needed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state from a saved state."""
        from_pickle = self.env.set_state(state)
        # Restore any additional state if needed
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on offensive strategies."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        # Iterate over each agent's observation
        for i in range(len(reward)):
            o = observation[i]
            
            # Reward for shooting at goal
            if o['game_mode'] in [6]:  # Assuming 6 indicates a shooting opportunity, like a penalty
                components['shooting_reward'][i] = self.shooting_reward

            # Reward for effective dribbling
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and o['sticky_actions'][9] == 1:  # 9 is dribbling
                components['dribbling_reward'][i] = self.dribbling_reward

            # Reward for passing, especially long or high passes
            if 'game_mode' in o and o['game_mode'] in [4, 5]:  # Assuming 4, 5 can be long/high passes modes
                components['passing_reward'][i] = self.passing_reward

            # Aggregate the total reward with added components
            reward[i] += (
                components['shooting_reward'][i] +
                components['dribbling_reward'][i] +
                components['passing_reward'][i]
            )

        return reward, components

    def step(self, action):
        """Wrap the environment's step function to include reward modification."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        
        # Update info with sticky actions counts
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action
        
        return observation, reward, done, info
