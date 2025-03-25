import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for strong defensive actions such as blocking and man-marking."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the sticky actions counter and internal state for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Extract state information for serialization."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from deserialized data."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Augment the reward based on defensive metrics."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Add rewards for controlling opposing players' movements
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:  # If opposing team has the ball
                controlled_opponents = sum([1 for pos in o['right_team'] if np.linalg.norm(pos - o['left_team'][o['active']]) < 0.1])
                components["defensive_reward"][rew_index] = 0.05 * controlled_opponents

            # Increase reward for blocking shots on goal
            if o['game_mode'] in {2, 3} and o['left_team'][o['active']][0] > 0.9:  # Near own goal in critical modes
                components["defensive_reward"][rew_index] += 0.2
            
            reward[rew_index] += components["defensive_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        """Processes the agent's action, updates the environment, and adjusts the reward."""
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
