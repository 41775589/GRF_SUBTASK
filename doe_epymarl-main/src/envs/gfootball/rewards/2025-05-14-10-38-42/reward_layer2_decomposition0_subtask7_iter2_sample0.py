import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances rewards for mastering 'Stop-Moving' strategies, focusing on position holding
    and interception. It rewards agents for sharp stopping closest to key strategic positions, mimicking
    ideal defensive coverage or interception logic.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_accuracy_threshold = 0.05  # Threshold for position accuracy to reward stop action
        self.stop_moving_reward = 0.1  # Reward for stopping near key positions
        self.key_positions = np.array([[0, 0], [0.5, 0], [-0.5, 0]])  # Key positions on the field

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
        """
        Modify reward based on the agent's ability to stop moving close to the key strategic positions, encouraging
        effective positioning and interception gameplay.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "stop_moving_positional_reward": [0.0]  # Initialize the reward for stopping near key positions
        }

        if observation is None:
            return reward, components

        o = observation[0]
        player_pos = o['left_team'][o['active']]
        for key_pos in self.key_positions:
            distance = np.linalg.norm(player_pos - key_pos)
            if o['sticky_actions'][0] == 1 and o['sticky_actions'][4] == 0 and distance < self.positional_accuracy_threshold:
                components["stop_moving_positional_reward"][0] = self.stop_moving_reward
                break

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
