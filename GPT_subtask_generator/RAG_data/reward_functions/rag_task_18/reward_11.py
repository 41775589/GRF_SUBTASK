import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing central midfield gameplay by rewarding controlled pace management and seamless transitions."""

    def __init__(self, env):
        super().__init__(env)
        self._num_zones = 5  # Number of field zones to consider
        self.zone_thresholds = np.linspace(-1, 1, self._num_zones + 1)  # Horizontal divisions of the field
        self.pace_rewards = np.linspace(0.1, 0.5, self._num_zones)  # Increasing reward for controlled pacing towards the center
        self.transition_reward = 0.1
        self.previous_active_players = None

    def reset(self):
        """Reset the environment and related variables."""
        self.previous_active_players = None
        return self.env.reset()

    def reward(self, reward):
        """Modify the reward for player actions that promote controlled pacing and effective transitions."""
        observation = self.env.unwrapped.observation()
        
        # Initialize reward components
        components = {"base_score_reward": reward.copy(),
                      "paced_control_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}
        
        # Apply zone and transition rewards
        if observation is not None:
            cur_active_players = []
            for index, o in enumerate(observation):
                player_x = o['right_team'][o['active']][0] if o['active'] != -1 else None
                if player_x is not None:
                    # Determine current field zone of the active player
                    zone_index = np.digitize([player_x], self.zone_thresholds) - 1
                    valid_zone = (zone_index >= 0) and (zone_index < self._num_zones)
                    if valid_zone:
                        components["paced_control_reward"][index] = self.pace_rewards[zone_index[0]]
                        
                    cur_active_players.append(o['active'])
            
            # Reward transitions if the active player has changed effectively
            if self.previous_active_players is not None:
                for i, (prev, cur) in enumerate(zip(self.previous_active_players, cur_active_players)):
                    if prev != cur and cur is not None:
                        components["transition_reward"][i] = self.transition_reward
            
            self.previous_active_players = cur_active_players
        
        # Combine components into total reward
        total_rewards = []
        for idx in range(len(reward)):
            total_reward = (reward[idx] +
                            components["paced_control_reward"][idx] +
                            components["transition_reward"][idx])
            total_rewards.append(total_reward)
        
        return total_rewards, components

    def step(self, action):
        """Step environment and add reward components to info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
