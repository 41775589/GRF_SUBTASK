import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive coordination reward specifically tailored 
    for enhancing defensive role understanding and positioning.”

    def __init__(self, env):
        super().__init__(env)
        self.defensive_zones = np.linspace(-1, 0, 5)  # Defines 5 zones in defensive half
        self.zone_rewards = np.zeros(5, dtype=int)  # Tracks rewards given in each zone
        self.penalty_area_reward = 0.3
        self.general_defensive_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.zone_rewards.fill(0)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.zone_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.zone_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_zone_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team'][o['designated']]
            
            in_penalty_area = (player_pos[0] < 0.6) and (abs(player_pos[1]) < 0.2)

            # Assign rewards based on zones
            zone_index = np.searchsorted(self.defensive_zones, player_pos[0], side='right') - 1
            if zone_index >= 0 and self.zone_rewards[zone_index] == 0:
                if in_penalty_area:
                    components["defensive_zone_reward"][rew_index] = self.penalty_area_reward
                else:
                    components["defensive_zone_reward"][rew_index] = self.general_defensive_reward
                self.zone_rewards[zone_index] = 1

            reward[rew_index] += components["defensive_zone_reward"][rew_index]

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
