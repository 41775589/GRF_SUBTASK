import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds an advanced strategic and defensive auxiliary reward for midfield and defensive mastery."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_zones_collected = {}
        self.midfield_control_rewards = {}
        self.defensive_reward_multiplier = 0.2
        self.control_reward_multiplier = 0.15

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_zones_collected = {}
        self.midfield_control_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_zones_collected'] = self.defensive_zones_collected
        to_pickle['midfield_control_rewards'] = self.midfield_control_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_zones_collected = from_pickle['defensive_zones_collected']
        self.midfield_control_rewards = from_pickle['midfield_control_rewards']
        return from_pickle

    def reward(self, reward):
        """Additional tactical reward for zone control and effective defensive maneuvers."""
        observation = self.env.unwrapped.observation()
        if observation is None:
          return reward, {'base_score_reward': reward}

        components = {
            "base_score_reward": reward.copy(),
            "defensive_zone_reward": [0.0] * len(reward),
            "midfield_control_reward": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team'][o['active']]
            
            # Defensive zone reward based on player's position
            if player_pos[0] < -0.7:  # Deep in own half
                if rew_index not in self.defensive_zones_collected:
                    components["defensive_zone_reward"][rew_index] = self.defensive_reward_multiplier
                    reward[rew_index] += components["defensive_zone_reward"][rew_index]
                    self.defensive_zones_collected[rew_index] = True
            
            # Midfield control reward: encouraging maintaining position in the midfield
            if -0.2 <= player_pos[0] <= 0.2:  # Midfield region
                if rew_index not in self.midfield_control_rewards:
                    components["midfield_control_reward"][rew_index] = self.control_reward_multiplier
                    reward[rew_index] += components["midfield_control_reward"][rew_index]
                    self.midfield_control_rewards[rew_index] = True

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
