import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a density reward focusing on developing winger crossing abilities."""

    def __init__(self, env):
        super().__init__(env)
        self.crossing_reward = 0.5
        self.sprinting_reward = 0.3
        self._crossings_made = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._crossings_made = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "crossing_reward": 0.0,
                      "sprinting_reward": 0.0}
        if observation is None:
            return reward, components

        # Player crosses the ball near the opponent's end of the field
        ball_position = observation.get('ball', [0, 0])
        if ball_position[0] > 0.75:  # Assume crossing if the ball is at the opponent's last quarter
            controlling_player_idx = observation.get('ball_owned_player', None)
            if controlling_player_idx in observation.get('left_team_roles', []) and \
               observation['left_team_roles'][controlling_player_idx] == 7:  # 7 is a winger role id
                components['crossing_reward'] += self.crossing_reward
                self._crossings_made += 1

        # Player is sprinting effectively towards the opponent's end
        controlling_player_idx = observation.get('active', None)
        player_position = observation.get('left_team', np.array([]))[controlling_player_idx]
        player_direction = observation.get('left_team_direction', np.array([]))[controlling_player_idx]
        if player_position[0] > 0.5 and player_direction[0] > 0.1:  # Moving forwards in the last half towards opponent's goal
            if observation['sticky_actions'][8] == 1:  # Assuming index 8 is the sprint action
                components['sprinting_reward'] += self.sprinting_reward

        # Aggregate rewards
        reward += components['crossing_reward'] + components['sprinting_reward']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        new_reward, components = self.reward(reward)
        info["final_reward"] = new_reward
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[action] += 1
        info.update({f"sticky_actions_{i}": count for i, count in enumerate(self.sticky_actions_counter)})
        return observation, new_reward, done, info
