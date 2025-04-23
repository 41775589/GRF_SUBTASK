import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for executing accurate long passes between 
    specified zones in the football pitch, focusing on distance, vision, 
    and precision of the pass.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_start_zone = None  # Position where the pass started
        self.is_pass_started = False
        
        # Define zones as tuples in the format (x_min, x_max, y_min, y_max)
        self.pass_zones = {
            "defense": (-1.0, -0.33, -0.42, 0.42),
            "midfield": (-0.33, 0.33, -0.42, 0.42),
            "attack": (0.33, 1.0, -0.42, 0.42)
        }

        # Coefficient for rewarding long passes
        self.long_pass_coefficient = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_start_zone = None
        self.is_pass_started = False
        return self.env.reset()
    
    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        # We extract observations directly from the environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['right_team'][o['active']]
            
            if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                if not self.is_pass_started:
                    # Mark the start of the pass
                    self.is_pass_started = True
                    self.pass_start_zone = player_pos
                else:
                    # Check if the ball is still with the same player
                    current_zone = self.identify_zone(player_pos)
                    start_zone = self.identify_zone(self.pass_start_zone)
                    
                    # If the current zone is different from the start zone and pass continues
                    if current_zone and start_zone and current_zone != start_zone:
                        # Reward only significant passes between different zones
                        components["long_pass_reward"][rew_index] = self.long_pass_coefficient
                        reward[rew_index] += components["long_pass_reward"][rew_index]
            else:
                # Reset pass tracking
                self.is_pass_started = False
                self.pass_start_zone = None

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

    def identify_zone(self, position):
        """
        Identifies the zone based on the player's position
        """
        for name, bounds in self.pass_zones.items():
            x_min, x_max, y_min, y_max = bounds
            if x_min <= position[0] <= x_max and y_min <= position[1] <= y_max:
                return name
        return None
