import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for performing high passes and crosses from various distances and angles."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_quality_multiplier = 0.1
        self.cross_quality_multiplier = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_threshold = 0.3  # Threshold for considering a pass 'high quality'
        self.cross_threshold = 0.5  # Threshold for considering a cross 'high quality'

    def reset(self):
        """Reset the environment and clear the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state for saving the environment's state."""
        pickle_state = self.env.get_state(to_pickle)
        pickle_state['sticky_actions_counter'] = self.sticky_actions_counter
        return pickle_state

    def set_state(self, state):
        """Set the environment's state."""
        state = self.env.set_state(state)
        self.sticky_actions_counter = state['sticky_actions_counter']
        return state

    def reward(self, reward):
        """Custom reward function to encourage high passes and effective crosses."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward
        
        # Initialize reward components
        components = {"base_score_reward": reward.copy(), "high_pass_reward": [0.0] * len(reward), "high_cross_reward": [0.0] * len(reward)}
        
        # Calculate rewards for high passes and crosses
        for i in range(len(reward)):
            player_obs = observation[i]
            ball_pos = player_obs['ball']
            ball_dir = player_obs['ball_direction']
            distance = np.linalg.norm([ball_pos[0], ball_pos[1]])

            # Check if it's a pass and calculate quality based on distance
            if player_obs['game_mode'] in [2, 3]:  # Assuming game_modes 2 or 3 indicate passes or crosses
                # If the pass/cross leads to positional advantage
                if distance > self.pass_threshold and np.abs(ball_dir[0]) > np.abs(ball_dir[1]):  # More forward than sideways
                    components["high_pass_reward"][i] += distance * self.pass_quality_multiplier
                elif distance > self.cross_threshold and np.abs(ball_dir[1]) > np.abs(ball_dir[0]):  # More sideways than forward
                    components["high_cross_reward"][i] += distance * self.cross_quality_multiplier
            
            # Accumulate customized rewards into the original reward list
            reward[i] += components["high_pass_reward"][i] + components["high_cross_reward"][i]

        return reward, components

    def step(self, action):
        """Take a step using the given action and apply the custom reward function."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Update info dictionary with reward breakdown
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Count sticky actions for informational purposes
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
