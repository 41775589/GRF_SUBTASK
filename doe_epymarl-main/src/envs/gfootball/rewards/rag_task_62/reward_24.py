import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for finishing techniques including optimal shooting angles and timing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shoot_angle_threshold = 0.1  # Define the angle within which the reward is given for shooting
        self.goal_position = [1, 0]  # X position of the goal is at 1, y position roughly at 0 center
        self.angle_reward = 1.0  # Reward for acceptable shooting angle
        self.timing_reward = 0.5  # Additional reward if shot taken under pressure
        self.pressing_defenders_threshold = 3  # Number of defenders considered as "pressure"

    def reset(self):
        """Reset sticky action counts and environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment for serialization."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the environment from serialization."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Compute and return the augmenting finishing technique reward based on the input reward list."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "timing_reward": [0.0] * len(reward),
                      "angle_reward": [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball'][:2]  # Extract x, y positions
            ball_direction = o['ball_direction'][:2]
            goal_direction = np.array(self.goal_position) - np.array(ball_pos)
            
            if np.linalg.norm(goal_direction) > 0 and np.linalg.norm(ball_direction) > 0:
                # Normalize vectors to compute dot product
                goal_direction_normalized = goal_direction / np.linalg.norm(goal_direction)
                ball_direction_normalized = ball_direction / np.linalg.norm(ball_direction)
                angle = np.dot(goal_direction_normalized, ball_direction_normalized)
                
                # Angle reward if the shot is towards goal within threshold
                if angle > 1 - self.shoot_angle_threshold:
                    components["angle_reward"][rew_index] = self.angle_reward
                    reward[rew_index] += self.angle_reward
                    
            # Timing reward if shot under pressure
            num_defenders_close = sum([np.linalg.norm(np.array(p) - np.array(ball_pos)) < self.shoot_angle_threshold 
                                       for p in o['right_team']])
            if num_defenders_close >= self.pressing_defenders_threshold:
                components["timing_reward"][rew_index] += self.timing_reward
                reward[rew_index] += self.timing_reward
        
        return reward, components

    def step(self, action):
        """Step function for the environment while keeping track of the reward and observation."""
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
