import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for effectively clearing the ball from defensive zones under pressure."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the regions considered as "defensive zones"
        self.defensive_threshold = -0.5  # X coordinate threshold to define defensive regions
        self.clearance_reward = 0.2       # Reward for effective clearance from defensive zone

    def reset(self):
        """Reset the sticky actions counter and environment on reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state including the state of this reward wrapper."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from a saved state, including the state of this reward wrapper."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on the player's ability to clear the ball from defensive zones."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if a clearance occurred: the ball transitions from inside to outside defensive zone
            ball_position = o['ball'][0]  # Get x-coordinate of the ball
            ball_owned_team = o['ball_owned_team']
            game_mode = o['game_mode']

            if ball_owned_team == 0 and ball_position < self.defensive_threshold:  # Left team is the agent team and ball in defensive zone
                if game_mode == 3 or game_mode == 5:  # FreeKick or ThrowIn implies successful clearance made
                    components["clearance_reward"][rew_index] = self.clearance_reward
                    reward[rew_index] += components["clearance_reward"][rew_index]
                
        return reward, components

    def step(self, action):
        """Take a step in the environment, returning the necessary information."""
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
