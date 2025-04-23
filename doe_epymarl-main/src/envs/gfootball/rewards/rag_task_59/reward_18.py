import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the coordination and decision-making skills of the goalkeeper under high-pressure scenarios."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Tracking previous positions to identify effective clearances
        self.previous_ball_position = None
        self.previous_keeper_position = None
        # Reward multipliers
        self.clearance_reward = 1.0
        self.positioning_reward = 0.1
        self.catch_penalty = -0.5

    def reset(self):
        """Reset the environment state and reward tracking."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        self.previous_keeper_position = None
        return self.env.reset()

    def reward(self, reward):
        """Modify the reward based on the goalkeeper's actions and effectiveness."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
        
        components = {'base_score_reward': reward.copy(),
                      'clearance_reward': 0.0,
                      'positioning_reward': 0.0,
                      'catch_penalty': 0.0}

        for index, o in enumerate(observation):
            ball_pos = np.array(o['ball'][:2])
            keeper_pos = o['left_team'][int(o['designated'])] if o['ball_owned_team'] == 0 else o['right_team'][int(o['designated'])]
            
            # Check if the goalkeeper has effectively cleared the ball
            if self.previous_ball_position is not None:
                distance_moved = np.linalg.norm(ball_pos - self.previous_ball_position)
                distance_from_goal = np.abs(keeper_pos[0])
                if distance_moved > 0.1 and distance_from_goal > 0.8:
                    components['clearance_reward'] += self.clearance_reward
            
            # Reward positioning - staying close to the goal line when under pressure
            if np.abs(keeper_pos[0]) > 0.85:
                components['positioning_reward'] += self.positioning_reward
            
            # Penalty for catching the ball (to encourage effective clearances)
            if o['game_mode'] == 6:  # Penalty kick mode, indicating a catch
                components['catch_penalty'] += self.catch_penalty
            
            # Combine all components to form the final reward
            reward[index] += components['clearance_reward'] + components['positioning_reward'] + components['catch_penalty']
            
            # Update previous states
            self.previous_ball_position = ball_pos.copy()
            self.previous_keeper_position = keeper_pos.copy()

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
