import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds tailored rewards for practicing shooting skills."""

    def __init__(self, env):
        super().__init__(env)
        self.shot_attempts = 0
        self.shot_targets = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and the reward wrapper parameters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_attempts = 0
        self.shot_targets = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Stores object state into pickle format."""
        to_pickle['shot_attempts'] = self.shot_attempts
        to_pickle['shot_targets'] = self.shot_targets
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores object state from pickle format."""
        from_pickle = self.env.set_state(state)
        self.shot_attempts = from_pickle['shot_attempts']
        self.shot_targets = from_pickle['shot_targets']
        return from_pickle

    def reward(self, reward):
        """Rewards the agent for practicing shots effectively."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_accuracy": [0.0],
                      "power": [0.0]}

        # Simple mechanism to detect shot by looking at the action space
        if observation['sticky_actions'][9] == 1:  # Assuming index 9 is the shoot action
            self.shot_attempts += 1
            # Reward for attempting a shot
            reward += 0.1
            components["power"][0] = 0.1

        # Check if the ball is towards the goal direction (simple heuristic)
        ball_direction = observation['ball_direction']
        if ball_direction[0] > 0.05:  # Assuming positive x-direction is towards opponent's goal
            self.shot_targets += 1
            # Increase reward if the shot is directed towards the goal
            reward += 0.2
            components["shooting_accuracy"][0] = 0.2

        return reward, components

    def step(self, action):
        """Performs an action in the environment, computes reward, and returns observation."""
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
