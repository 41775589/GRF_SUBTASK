import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive maneuvers and midfield control."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Initialize custom tracking variables
        self._num_defensive_zones = 5
        self._num_midfield_zones = 5
        self.defensive_rewards_collected = {}
        self.midfield_rewards_collected = {}
        self.defensive_zone_threshold = 0.5
        self.midfield_zone_threshold = 0.1
        # Coefficients for additional rewards
        self.defensive_reward_coefficient = 0.2
        self.midfield_reward_coefficient = 0.15
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_rewards_collected = {}
        self.midfield_rewards_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'defensive_rewards_collected': self.defensive_rewards_collected,
            'midfield_rewards_collected': self.midfield_rewards_collected,
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_rewards_collected = from_pickle['CheckpointRewardWrapper']['defensive_rewards_collected']
        self.midfield_rewards_collected = from_pickle['CheckpointRewardWrapper']['midfield_rewards_collected']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        # Initialize reward components structure
        components = {
            "base_score_reward": reward.copy(),
            "defensive_zone_reward": [0.0] * len(reward),
            "midfield_control_reward": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            ball_pos = o['ball'][:2]

            # Defensive reward: if active player is in own half and closer to own goal than the opponent, add reward
            if player_pos[0] * np.sign(ball_pos[0]) < self.defensive_zone_threshold:
                if rew_index not in self.defensive_rewards_collected:
                    components["defensive_zone_reward"][rew_index] = self.defensive_reward_coefficient
                    reward[rew_index] += components["defensive_zone_reward"][rew_index]
                    self.defensive_rewards_collected[rew_index] = True

            # Midfield Control: reward positive interactions in the midfield zone
            if abs(player_pos[0]) < self.midfield_zone_threshold:
                if rew_index not in self.midfield_rewards_collected:
                    components["midfield_control_reward"][rew_index] = self.midfield_reward_coefficient
                    reward[rew_index] += components["midfield_control_reward"][rew_index]
                    self.midfield_rewards_collected[rew_index] = True

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()

        # Update the sticky actions counter
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
