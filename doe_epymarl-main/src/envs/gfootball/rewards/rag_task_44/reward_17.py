import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards for effective dribble-stop maneuvers under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize settings for dribble-stop behavior rewards.
        self.stop_dribble_reward = 0.1
        self.pressure_threshold = 0.01  # Simulated 'pressure' metric threshold

    def reset(self):
        """Reset the environment and the sticky action counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state to be pickled, along with additional state components."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from unpickled state and restore additional components."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Modify the reward to encourage dribble-stop under opposing pressure."""
        
        # Get current observation from the environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_dribble_reward": [0.0 for _ in reward]}

        # Ensure the observation is populated
        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            # Calculate simulated 'pressure' based on opponent proximity and ball control
            ball_pos = obs['ball']
            player_pos = obs['left_team'][obs['active']] if obs['ball_owned_team'] == 0 else obs['right_team'][obs['active']]
            opponent_team = 'right_team' if obs['ball_owned_team'] == 0 else 'left_team'
            closest_opponent_distance = min(np.linalg.norm(player_pos - player) for player in obs[opponent_team])

            # Check for stop_dribble action (8 corresponds to dribble in sticky actions)
            stop_dribble_action = (obs['sticky_actions'][8] == 0)

            # Reward if stopping dribbling under pressure
            if stop_dribble_action and closest_opponent_distance < self.pressure_threshold:
                components["stop_dribble_reward"][i] = self.stop_dribble_reward
                reward[i] += components["stop_dribble_reward"][i]

        return reward, components

    def step(self, action):
        """Take a step in the environment, handle the reward, and return modified observations and rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
