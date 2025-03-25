import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on defensive teamwork near the penalty area."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions = []
        self.num_zones = 2  # Number of zones in the penalty area
        self.zone_control_rewards = [0.1, 0.15]  # Reward for controlling each subsequent zone
        self.max_defensive_reward = sum(self.zone_control_rewards)
        self._initialize_defensive_zones()

    def _initialize_defensive_zones(self):
        # Example positions could be dynamically calculated or pre-defined
        self.defensive_positions = [
            (-0.9, 0.2),   # Left side of penalty area
            (-0.9, -0.2)   # Right side of penalty area
        ]

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_positions'] = self.defensive_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_positions = from_pickle['defensive_positions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_zone_control": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            left_team_positions = o['left_team']
            ball_position = o['ball'][:2]  # ignore z-coordinate

            for index, (zone_x, zone_y) in enumerate(self.defensive_positions):
                for player_pos in left_team_positions:
                    # Calculate Euclidean distance from player to zone
                    distance_to_zone = np.sqrt((player_pos[0] - zone_x)**2 + (player_pos[1] - zone_y)**2)
                    if distance_to_zone < 0.05:  # Threshold distance to consider the zone controlled
                        components['defensive_zone_control'][rew_index] += self.zone_control_rewards[index]
                        break
        
        # Update the final rewards considering the defensive actions
        for rew_index in range(len(reward)):
            reward[rew_index] += components['defensive_zone_control'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info["component_" + key] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
