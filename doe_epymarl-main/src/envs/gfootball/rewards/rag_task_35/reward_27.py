import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes strategic positioning and movement,
    encouraging effective pivoting between defensive placements and attacking.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward scaling for strategic positioning
        self.defensive_bonus = 0.1
        self.attack_prep_bonus = 0.1
        # Positions defining strategic zones, these should be set according to game specifics
        self.defensive_zones = [-1.0, -0.5]  # x positions considered as defensive
        self.attack_prep_zones = [0.5, 1.0]  # x positions to prepare attacks

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_bonus": [0.0] * len(reward),
                      "attack_prep_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos_x = o['left_team'][o['active']][0] if o['ball_owned_team'] == 0 else o['right_team'][o['active']][0]

            # Defensive strategy: good positioning in defensive zone
            if player_pos_x < self.defensive_zones[1]:
                components["defensive_bonus"][rew_index] = self.defensive_bonus
                reward[rew_index] += components["defensive_bonus"][rew_index]

            # Preparing attacks: moving towards attack zones
            if player_pos_x > self.attack_prep_zones[0]:
                components["attack_prep_bonus"][rew_index] = self.attack_prep_bonus
                reward[rew_index] += components["attack_prep_bonus"][rew_index]

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
