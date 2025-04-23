import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that focuses on defensive skills like responsiveness and interception in high-pressure scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions = 0
        self.defensive_positions = 0
        self.interception_reward = 0.5
        self.positioning_reward = 0.3

    def reset(self):
        """Resets the environment and the associated counters for sticky actions, interceptions, and defensive positions."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions = 0
        self.defensive_positions = 0
        return self.env.reset()

    def reward(self, reward):
        """Augments the reward based on defensive skills such as interceptions and smart defensive positioning."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        components = {
            'base_score_reward': reward,
            'interception_reward': 0.0,
            'positioning_reward': 0.0,
        }

        for o in observation:
            if o['game_mode'] == 0:  # Check for normal game mode
                # Encourage intercepting the ball if an opponent owns it
                if o['ball_owned_team'] == 1:  # Opponent owns the ball
                    if np.random.random() < 0.1:  # Simulate an interception event
                        self.interceptions += 1
                        reward += self.interception_reward
                        components['interception_reward'] = self.interception_reward

                # Reward positional play within a critical defensive area
                defensive_zone = [(-1, -0.3), (-1, 0.3)]
                player_pos = o['right_team'][o['active']]  # Active player on the right team (defensive)
                if defensive_zone[0][0] < player_pos[0] < defensive_zone[1][0] \
                   and defensive_zone[0][1] < player_pos[1] < defensive_zone[1][1]:
                    self.defensive_positions += 1
                    reward += self.positioning_reward
                    components['positioning_reward'] = self.positioning_reward

        return reward, components

    def step(self, action):
        """Executes a step in the environment, applying the custom reward adjustments and tracking sticky actions."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = reward
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
