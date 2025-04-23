import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward encouraging defensive coordination and efficient transitions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.num_defensive_zones = 5
        self.zone_rewards = np.linspace(0.05, 0.2, self.num_defensive_zones)
        self.collected_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.collected_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.collected_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.collected_rewards = from_pickle.get('CheckpointRewardWrapper', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_coordination_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o.get('ball')
            ball_owned_team = o.get('ball_owned_team')
            
            if ball_owned_team != 0:  # Focus on the team 0 defensive rewards
                continue

            player_pos = o.get('left_team')
            opponent_pos = o.get('right_team')

            # Calculate the closest opponent to the ball (enemy closest to our team's defense)
            if ball_position is not None and opponent_pos is not None:
                opponent_distances = np.linalg.norm(opponent_pos - ball_position[:2], axis=1)
                closest_opponent = np.min(opponent_distances)
                
                # Detect which zone this falls into, larger distances contribute to higher rewards
                zone_thresholds = np.linspace(np.max(opponent_distances), np.min(opponent_distances), self.num_defensive_zones + 1)
                zone_index = np.digitize([closest_opponent], zone_thresholds) - 1

                if zone_index >= 0 and zone_index < self.num_defensive_zones:
                    if rew_index not in self.collected_rewards:
                        self.collected_rewards[rew_index] = set()

                    if zone_index not in self.collected_rewards[rew_index]:
                        reward[rew_index] += self.zone_rewards[zone_index]
                        components["defensive_coordination_reward"][rew_index] = self.zone_rewards[zone_index]
                        self.collected_rewards[rew_index].add(zone_index)

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
