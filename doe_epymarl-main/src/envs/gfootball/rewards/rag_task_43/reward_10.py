import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for defensive tasks."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize the counter for defensive positions
        self.defensive_positions_collected = {}
        self.num_defensive_zones = 8
        self.defensive_reward = 0.2

    def reset(self):
        """Resets the environment and the tracking variables."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Returns the state with defensive positions included."""
        to_pickle['defensive_positions_collected'] = self.defensive_positions_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the state from pickle and includes defensive positions."""
        from_pickle = self.env.set_state(state)
        self.defensive_positions_collected = from_pickle['defensive_positions_collected']
        return from_pickle

    def reward(self, reward):
        """Customizes the reward function to encourage defensive strategies."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for defensive positioning and intercepting passes
            if ('ball_owned_team' in o and o['ball_owned_team'] == 0): # Assuming 0 is the opponent
                defensive_zone_index = int(o['ball'][0] * self.num_defensive_zones)
                if defensive_zone_index not in self.defensive_positions_collected:
                    components["defensive_reward"][rew_index] = self.defensive_reward
                    reward[rew_index] += self.defensive_reward
                    self.defensive_positions_collected[defensive_zone_index] = 1

            # Adjust reward based on ball possession changes
            if o['ball_owned_team'] == 1:  # Assuming 1 is the controlled team
                reward[rew_index] += 0.1  # small reward for gaining possession
            elif o['ball_owned_team'] == -1:  # No team controls the ball
                reward[rew_index] -= 0.05  # small penalty for losing possession

        return reward, components

    def step(self, action):
        """Processes the environment step and augments it with detailed reward calculations."""
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
