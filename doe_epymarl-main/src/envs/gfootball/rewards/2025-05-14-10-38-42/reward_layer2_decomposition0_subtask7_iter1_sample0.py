import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards for mastering 'Stop-Moving' strategies, focusing on position holding and interception."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_accuracy_threshold = 0.05  # Threshold for position accuracy to reward stop action

    def reset(self):
        """Reset the sticky action counters and environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state with the inclusion of sticky actions count."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from stored information including sticky actions count."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Modify reward based on the agent's ability to stop moving close to the ball accurately."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),  # Store original reward
            "stop_moving_positional_reward": [0.0]  # Initialize the reward for stopping moving near optimal position
        }

        if observation is None:
            return reward, components

        o = observation[0]
        player_pos = o['left_team'][o['active']]
        ball_pos = o['ball'][:2]
        distance_to_ball = np.linalg.norm(ball_pos - player_pos)

        if o['sticky_actions'][7] == 1 and distance_to_ball < self.positional_accuracy_threshold:
            components["stop_moving_positional_reward"][0] = 1.0 - distance_to_ball
        elif o['sticky_actions'][7] == 1:
            components["stop_moving_positional_reward"][0] = -0.5 * distance_to_ball

        reward[0] += components["stop_moving_positional_reward"][0]

        return reward, components

    def step(self, action):
        """Execute a step in the environment while modifying rewards."""
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
