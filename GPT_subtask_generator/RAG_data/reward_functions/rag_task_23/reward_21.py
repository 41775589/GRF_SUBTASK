import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.defensive_zones = np.linspace(-1, 1, num=10)  # 10 defensive zones across x-axis
        self.zone_visits = set()  # Tracks which zones have been visited
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.zone_visits = set()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'defensive_zones': list(self.defensive_zones),
                                                'zone_visits': list(self.zone_visits)}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        self.defensive_zones = np.array(from_picle['CheckpointRewardWrapper']['defensive_zones'])
        self.zone_visits = set(from_picle['CheckpointRewardWrapper']['zone_visits'])
        return from_picle

    def reward(self, rewards):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": rewards.copy(), "defensive_coverage_reward": [0.0] * len(rewards)}
        
        if observation is None:
            return rewards, components

        for i, (rew, obs) in enumerate(zip(rewards, observation)):
            current_pos = obs['left_team'][obs['active']][:2]  # Active player's position
            for zone in self.defensive_zones:
                # Check if player's x-coordinate is within this zone and zone not visited
                if zone - 0.1 < current_pos[0] <= zone + 0.1 and zone not in self.zone_visits:
                    self.zone_visits.add(zone)
                    components["defensive_coverage_reward"][i] += 0.5  # Reward for covering this zone

            # Update reward
            rewards[i] += components["defensive_coverage_reward"][i]

        return rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action_value in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action_value
        return observation, reward, done, info
