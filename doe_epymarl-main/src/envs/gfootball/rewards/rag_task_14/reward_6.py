import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a specific reward for the sweeper role in a football game, focusing on clearing the ball from
    the defensive zone, tackling, and fast recoveries.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._successful_clearances = 0
        self._tackles_made = 0
        self._fast_recoveries = 0

        # Rewards for each successful action by the sweeper
        self._clearance_reward = 0.3
        self._tackle_reward = 0.5
        self._recovery_reward = 0.2

    def reset(self):
        """
        Reset the environment and the counters for sweeper actions.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._successful_clearances = 0
        self._tackles_made = 0
        self._fast_recoveries = 0
        return self.env.reset()

    def reward(self, reward):
        """
        Modify the agent's reward based on successful sweeper-specific actions.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward,
            "clearance_reward": 0.0,
            "tackle_reward": 0.0,
            "recovery_reward": 0.0
        }

        # Evaluate the sweeper's actions based on observation
        if observation is None:
            return reward, components

        active_player_position = observation['right_team'][observation['active']]

        # Reward for clearing the ball from a critical defensive position
        if (self._is_in_defensive_zone(active_player_position) and
                'ball_owned_player' in observation and observation['ball_owned_player'] == observation['active']):
            self._successful_clearances += 1
            reward += self._clearance_reward
            components["clearance_reward"] += self._clearance_reward

        # Reward for making a successful tackle
        if observation['game_mode'] in (3, 5):  # Free kick or Throw in might indicate a successful tackle event
            self._tackles_made += 1
            reward += self._tackle_reward
            components["tackle_reward"] += self._tackle_reward

        # Reward for fast recoveries to a defensive position
        if self._made_fast_recovery(active_player_position):
            self._fast_recoveries += 1
            reward += self._recovery_reward
            components["recovery_reward"] += self._recovery_reward

        return reward, components

    def step(self, action):
        """
        Step the environment and enhance the reward signal.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info

    def _is_in_defensive_zone(self, player_position):
        """
        Check if the player is in the defensive zone.
        """
        x_position = player_position[0]
        return x_position < 0  # Assuming negative x is the defensive side for the right team

    def _made_fast_recovery(self, player_position):
        """
        Placeholder heuristic for detecting fast recovery.
        Currently just checks if the player is moving towards the defensive zone.
        """
        x_velocity = self.env.unwrapped.observation()['right_team_direction'][self.env.unwrapped.observation()['active']][0]
        return x_velocity < 0  # Negative velocity towards the defensive goal
