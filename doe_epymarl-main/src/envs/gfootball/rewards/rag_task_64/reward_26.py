import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for high passes and crossing from varying distances and angles."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the number of zones for measuring crossing/passing efforts
        self.num_zones = 5
        self.passing_reward = np.linspace(0.1, 0.5, self.num_zones)  # Increasing reward based on distance
        self.crossing_rewards = {'high_pass': 0.3, 'low_pass': 0.1, 'cross': 0.4}
        self.last_ball_position = None

    def reset(self):
        """Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment including additional data from this wrapper."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment and update the sticky actions counter from the state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Calculate additional reward for successful high passes and crosses from varying distances and angles."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'high_pass_reward': [0.0, 0.0], 'cross_reward': [0.0, 0.0]}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for idx, obs in enumerate(observation):
            ball_pos = obs['ball'][:2]  # Get x, y coordinates of the ball
            ball_owner = obs['ball_owned_team']
            game_mode = obs['game_mode']

            high_pass_trigger = obs['sticky_actions'][8]  # Action for high pass
            if ball_owner == 1 and high_pass_trigger and self.last_ball_position is not None:
                # Calculate the distance the ball was passed
                distance = np.linalg.norm(np.array(ball_pos) - np.array(self.last_ball_position))
                # Determine the zone based on the distance
                zone_idx = min(int(distance * self.num_zones), self.num_zones - 1)
                components['high_pass_reward'][idx] = self.passing_reward[zone_idx]
                reward[idx] += components['high_pass_reward'][idx]
            
            # Check for crossing based on game mode
            if game_mode == 4 and ball_owner == 1:  # Assumption that '4' is a crossing play mode
                cross_type = 'high_pass' if high_pass_trigger else 'low_pass'
                components['cross_reward'][idx] = self.crossing_rewards[cross_type]
                reward[idx] += components['cross_reward'][idx]
            
            # Update last ball position
            self.last_ball_position = ball_pos.copy()

        return reward, components

    def step(self, action):
        """Perform a step using the wrapped environment, augment rewards, and manage sticky action counters."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_item in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_item
        return observation, reward, done, info
