import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a custom reward for promoting offensive strategies."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_reward = 0.5
        self.dribbling_reward = 0.3
        self.passing_reward = 0.2

    def reset(self):
        """Reset the state of the environment and counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the wrapper."""
        # Custom wrapper state can be saved here if needed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the wrapper."""
        # Custom wrapper state can be reloaded here if needed
        return self.env.set_state(state)

    def reward(self, reward):
        """Customize rewards to encourage shooting, dribbling, and passing effectively."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(), 
            "shooting_reward": [0.0] * len(reward), 
            "dribbling_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Encourage shooting when close to the goal and the ball is controlled
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and abs(o['ball'][0]) > 0.9:
                components["shooting_reward"][rew_index] = self.shooting_reward
            
            # Encourage dribbling by rewarding stick action movements while ball is controlled
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and sum(o['sticky_actions'][0:9]) > 0:
                components["dribbling_reward"][rew_index] = self.dribbling_reward

            # Encourage passing by detecting high and long balls triggered by actions
            if 'ball_direction' in o and (abs(o['ball_direction'][1]) > 0.1 or abs(o['ball_direction'][0]) > 0.5):
                components["passing_reward"][rew_index] = self.passing_reward

            # Aggregate custom rewards with the initial game reward
            reward[rew_index] += components["shooting_reward"][rew_index] + \
                                 components["dribbling_reward"][rew_index] + \
                                 components["passing_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Take an action using the wrapped environment and augment reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update info with current sticky actions counts
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
