import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing team synergy during possession changes, emphasizing
    precise timing and strategic positioning of both offensive and defensive moves."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_possession = -1  # Track the previous possession state
        self.possession_change_reward = 0.3  # Reward for successfully changing possession

    def reset(self):
        """Reset the reward wrapper state and environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_possession = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the current state of the wrapper."""
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'previous_possession': self.previous_possession
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the previously saved state of the wrapper."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        self.previous_possession = from_pickle['CheckpointRewardWrapper']['previous_possession']
        return from_pickle

    def reward(self, reward):
        """Modify the reward to emphasize successful possession changes and strategic positioning."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward}

        components = {"base_score_reward": reward.copy(), "possession_change_reward": [0.0, 0.0]}
        
        for i in range(len(observation)):
            o = observation[i]
            current_possession = o['ball_owned_team']
            
            # Reward for changing possession correctly
            if self.previous_possession != -1 and current_possession != self.previous_possession:
                if current_possession == i:  # Our agent just gained possession
                    components["possession_change_reward"][i] += self.possession_change_reward
                    reward[i] += components["possession_change_reward"][i]

        self.previous_possession = current_possession
        return reward, components

    def step(self, action):
        """Execute a step in the underlying environment."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Include reward components in information for debugging purposes
        for key, value in components.items():
            info["component_" + key] = sum(value)
        
        # Track sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        
        return observation, reward, done, info
