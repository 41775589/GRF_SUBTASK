import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for long-range shots and distance covered towards the goal from outside the box."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._long_shot_threshold = 0.7  # Shots beyond this X distance threshold
        self._distance_reward_scale = 0.01  # Scale factor for rewarding distance

    def reset(self):
        """Reset the environment and any necessary variable."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the current state of the environment, with wrapper specific values."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment with wrapper specific values."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify the rewards given by the environment."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "long_shot_reward": [0.0] * len(reward), "distance_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            ball_pos = o['ball'][0]  # Get the X position of the ball
            # Reward for long shots goals
            if o['game_mode'] == 6 and ball_pos > self._long_shot_threshold:  # Assuming mode 6 represents a shot
                if reward[rew_index] == 1:  # Check if a goal has been scored
                    components['long_shot_reward'][rew_index] = 1.0  # Additional reward for long-shot goals
                    
            # Reward for moving towards opponent's goal from afar
            if o['ball_owned_team'] == 1 and ball_pos > self._long_shot_threshold:
                opponent_goal_distance = 1.0 - ball_pos  # Distance from the right goal
                components['distance_reward'][rew_index] = opponent_goal_distance * self._distance_reward_scale

            # Combine all rewards components
            reward[rew_index] += components['long_shot_reward'][rew_index] + components['distance_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Step the environment, modify the reward, and provide additional info."""
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
