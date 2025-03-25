import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to focus on enhancing high-pass skills by targeting rewards for trajectory control, power assessment, and situational plays."""
    
    def __init__(self, env):
        super().__init__(env)
        self._num_zones = 5  # Number of zones to target passes
        self._zone_rewards = [0.2, 0.4, 0.6, 0.8, 1.0]  # Differential rewards for each zone of the pitch
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._passes_completed_in_zones = [0] * self._num_zones

    def reset(self):
        """Reset the internal state upon starting a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._passes_completed_in_zones = [0] * self._num_zones
        return self.env.reset()
        
    def get_state(self, to_pickle):
        """Retrieve state with specific information on completed passes in zones."""
        to_pickle['CheckpointRewardWrapper'] = self._passes_completed_in_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from external data, particularly for continue training or evaluation."""
        from_pickle = self.env.set_state(state)
        self._passes_completed_in_zones = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Calculate augmented reward based on the high-pass execution effectiveness in different pitch zones."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "high_pass_precision_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Loop through rewards and observations
        for rew_index, o in enumerate(observation):
            # We assume here o contains 'high_pass_success' (True/False) and 'zone_of_pass' (0-4)
            if o.get('high_pass_success', False):
                zone = o.get('zone_of_pass', 0)
                components["high_pass_precision_reward"][rew_index] = self._zone_rewards[zone]
                reward[rew_index] += self._zone_rewards[zone]
                self._passes_completed_in_zones[zone] += 1
                
        return reward, components

    def step(self, action):
        """Override step to include handling of custom reward calculation."""
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
