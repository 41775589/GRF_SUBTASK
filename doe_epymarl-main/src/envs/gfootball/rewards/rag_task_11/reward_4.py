import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that adds offensive training reinforcement through fast-paced maneuvers and precision-based finishing control. """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define number of zones from midfield to the opponent's goal
        self._num_zones = 5
        self._zone_rewards = np.linspace(0.1, 0.5, self._num_zones)
        self._last_zone_triggered = -1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._last_zone_triggered = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = {'last_zone_triggered': self._last_zone_triggered}
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._last_zone_triggered = from_pickle['CheckpointRewardWrapper']['last_zone_triggered']
        return from_pickle

    def reward(self, reward):
        # Access the current observation
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_training_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        ball_pos = observation[0]['ball'][0]  # Considering X-axis position of the ball
        player_pos = observation[0]['left_team'][observation[0]['active']][0]

        # Encourage offensive plays by giving rewards based on ball's advance towards the opponent's goal
        if ball_pos > player_pos:  # Ball is ahead of the player position
            current_zone = int(np.clip(np.floor(5 * (ball_pos + 1) / 2), 0, self._num_zones - 1))
            if current_zone > self._last_zone_triggered:
                zone_diff = current_zone - self._last_zone_triggered
                for i in range(zone_diff):
                    reward[0] += self._zone_rewards[current_zone - i]
                    components["offensive_training_reward"][0] += self._zone_rewards[current_zone - i]

                self._last_zone_triggered = current_zone

        return reward, components

    def step(self, action):
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
