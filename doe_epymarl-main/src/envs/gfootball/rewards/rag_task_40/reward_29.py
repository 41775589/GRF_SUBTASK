import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies rewards to improve defensive strategies and counterattack positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)

        # Initialize variables to track defensive checkpoints and ball proximity
        self.defensive_zones_occupied = set()
        self.previous_ball_distance = float('inf')
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Define reward weights for various defensive strategies
        self.defensive_position_reward = 0.05
        self.ball_proximity_reward = 0.1
        self.counterattack_setup_reward = 0.2

    def reset(self):
        """Reset the environment and clear the defensive tracking."""
        self.defensive_zones_occupied.clear()
        self.previous_ball_distance = float('inf')
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store state information for checkpoint tracking."""
        to_pickle['CheckpointRewardWrapper'] = {
            'defensive_zones_occupied': self.defensive_zones_occupied,
            'previous_ball_distance': self.previous_ball_distance
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state based on previously stored data."""
        from_pickle = self.env.set_state(state)
        saved_state = from_pickle['CheckpointRewardWrapper']
        self.defensive_zones_occupied = saved_state['defensive_zones_occupied']
        self.previous_ball_distance = saved_state['previous_ball_distance']
        return from_pickle

    def reward(self, reward):
        """Modify reward based on defensive performance and ball proximity."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}

        for rew_index, rew in enumerate(reward):
            o = observation[rew_index]
            components["defensive_reward"][rew_index] = 0

            # Defensive position reward: reward for each player positioned closely to own goal (defensive stance)
            defensive_position = -0.1 < o['left_team'][o['active']][0] < 0.1
            
            if defensive_position and o['active'] not in self.defensive_zones_occupied:
                components["defensive_reward"][rew_index] += self.defensive_position_reward
                self.defensive_zones_occupied.add(o['active'])

            # Reward for minimizing ball distance while in possession by opposing team, stimulating counterattack readiness
            ball_distance = np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']])
            if o['ball_owned_team'] == 1 and ball_distance < self.previous_ball_distance:
                components["defensive_reward"][rew_index] += self.ball_proximity_reward
                self.previous_ball_distance = ball_distance

            # Combine rewards
            reward[rew_index] += sum(components["defensive_reward"][rew_index])

        return reward, components

    def step(self, action):
        """Perform a step in the environment with additional reward modification."""
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
