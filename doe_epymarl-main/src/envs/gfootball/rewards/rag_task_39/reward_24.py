import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for clearing the ball effectively from defensive zones."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.safe_clearance_reward = 1.0  # Reward for clearing the ball safely under pressure
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_zones = {  # Define zones as X-axis intervals where clearances are rewarded
            'critical_zone': [-1.0, -0.8],  # Closer to the own goal
            'defensive_mid': [-0.8, -0.4]
        }

    def reset(self):
        """Reset the wrapper's state for a new game environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Modify the reward based on effective clearance in defensive pressure."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'clearance_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['game_mode'] == 0 and o['ball_owned_team'] == 0:  # Normal play and ball owned by left team
                ball_position = o['ball'][0]  # X-axis coordinates of the ball
                
                # Check if the clearance is made from defensive zones under pressure
                if any(lower <= ball_position <= upper for (lower, upper) in self.clearance_zones.values()):
                    if self._is_defensive_pressure(o):
                        components['clearance_reward'][rew_index] = self.safe_clearance_reward
                        reward[rew_index] += components['clearance_reward'][rew_index]
                        
        return reward, components

    def step(self, action):
        """Executes a step in the environment and processes the custom reward adjustment."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info

    def _is_defensive_pressure(self, o):
        """Estimates if there is defensive pressure based on proximity of opponents."""
        player_pos = o['left_team'][o['active']]  # Position of the player with the ball
        opponents_pos = o['right_team']
        # Check if any opponent is within a threatening distance
        return np.any([np.linalg.norm(player_pos - opp) < 0.15 for opp in opponents_pos])
