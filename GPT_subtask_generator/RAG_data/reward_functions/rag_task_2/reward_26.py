import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on defensive teamwork and strategic positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.defensive_zones = 5  # Number of defensive zones to monitor
        self.zone_threshold = 1 / self.defensive_zones
        self.agent_positions = []
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_zones = [False] * self.defensive_zones
        self.opponent_approach_zones = [False] * self.defensive_zones

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_zones = [False] * self.defensive_zones
        self.opponent_approach_zones = [False] * self.defensive_zones
        self.agent_positions = []
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'ball_control_zones': self.ball_control_zones,
            'opponent_approach_zones': self.opponent_approach_zones
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_info = from_pickle['CheckpointRewardWrapper']
        self.ball_control_zones = state_info['ball_control_zones']
        self.opponent_approach_zones = state_info['opponent_approach_zones']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            self.agent_positions.append(o['left_team'])
            distance_to_ball = np.linalg.norm(o['ball'][:2] - o['left_team'][o['active']][:2])
            zone_index = int(distance_to_ball // self.zone_threshold)

            # Reward defending the ball in different zones
            if zone_index < self.defensive_zones and not self.ball_control_zones[zone_index]:
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                    self.ball_control_zones[zone_index] = True
                    components["defensive_positioning_reward"][rew_index] += 0.1

            # Reward preventing opponent's approach in defensive zones 
            for opponent_position in o['right_team']:
                distance_to_goal = np.abs(opponent_position[0] + 1)  # right goal at x = 1
                opposition_zone_index = int(distance_to_goal // self.zone_threshold)
                if opposition_zone_index < self.defensive_zones and not self.opponent_approach_zones[opposition_zone_index]:
                    self.opponent_approach_zones[opposition_zone_index] = True
                    components["defensive_positioning_reward"][rew_index] += 0.05

                # Reinforce the reward by multiplying by how many zones are defended
                zones_controlled = sum(self.ball_control_zones)
                reward[rew_index] += 0.1 * zones_controlled
                components["defensive_positioning_reward"][rew_index] += 0.1 * zones_controlled
        
        for rew_index, rew in enumerate(reward):
            reward[rew_index] = components["base_score_reward"][rew_index] + components["defensive_positioning_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Adding component rewards to the info for debugging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Update sticky actions counter for debugging purposes
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
