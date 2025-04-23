import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the reward function to encourage shooting from central field positions with accuracy and power."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        """Reset the environment and clear sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment for serialization."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from deserialization."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify the reward to encourage accurate and powerful shots from central field positions."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),   # Include the original reward
            "shooting_position_reward": [0.0] * len(reward)  # Initialize a new reward component
        }
        
        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]
            ball_pos = o['ball'][:2]  # Get the x, y position of the ball
            ball_owner = (o['ball_owned_team'], o['ball_owned_player'])
            if ball_owner[0] in [0, 1]:  # Check if either team owns the ball
                active_player_team = 'left_team' if ball_owner[0] == 0 else 'right_team'
                active_player_idx = o[ball_owner[1]]
                active_player_pos = o[active_player_team][active_player_idx]

                # Assuming central field positions are within y-range of -0.1 to 0.1 near the center line
                if -0.1 <= active_player_pos[1] <= 0.1 and -0.3 <= active_player_pos[0] <= 0.3:
                    shot_power = 0.0  # Measure of shot strength (integrate with your game engine stats)
                    if shot_power > 0.5:  # Threshold for considering a shot powerful
                        components["shooting_position_reward"][i] = 0.5  # Reward for powerful and accurate shots
                        reward[i] += components["shooting_position_reward"][i]
        
        return reward, components

    def step(self, action):
        """Step environment and modify reward and other info based on the new reward scheme."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions counter for info
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                
        return observation, reward, done, info
