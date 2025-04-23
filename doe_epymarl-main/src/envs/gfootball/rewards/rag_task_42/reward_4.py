import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper for incorporating midfield dynamics mastering into rewards."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define midfield dominance by controlling zones dynamically based on game state
        self.midfield_zones = [0.2, 0.4, 0.6, 0.8]  # Midfield relative positions
        self.midfield_rewards = np.linspace(0.1, 0.4, len(self.midfield_zones))  # Rewards for each zone controlled
        self._last_controlled_position = {}
        self.midfield_control_reward = 0.05  # Additional reward for transferring control under pressure
        self.defense_to_offense_reward = 0.3  # Reward for transitioning from defense to offense
        self.offense_to_defense_reward = 0.3  # Reward for transitioning from offense to defense

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._last_controlled_position = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['midfield_control'] = self._last_controlled_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._last_controlled_position = from_pickle.get('midfield_control', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_zone_control": [0.0] * len(reward),
                      "dynamic_transition": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Calculating midfield control
            loc_x = o['ball'][0]  # Using ball x-position as control indicator

            # Determine current zone control
            current_zone_index = None
            for idx, pos in enumerate(self.midfield_zones):
                if loc_x < pos:
                    current_zone_index = idx
                    break

            if current_zone_index is not None:
                if rew_index in self._last_controlled_position:
                    if self._last_controlled_position[rew_index] != current_zone_index:
                        # Detecing transitions and rewarding accordingly
                        if self._last_controlled_position[rew_index] < current_zone_index:
                            components["dynamic_transition"][rew_index] = self.defense_to_offense_reward
                        else:
                            components["dynamic_transition"][rew_index] = self.offense_to_defense_reward
                        reward[rew_index] += components["dynamic_transition"][rew_index]
                reward[rew_index] += self.midfield_rewards[current_zone_index]
                components["midfield_zone_control"][rew_index] = self.midfield_rewards[current_zone_index]
                self._last_controlled_position[rew_index] = current_zone_index

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
