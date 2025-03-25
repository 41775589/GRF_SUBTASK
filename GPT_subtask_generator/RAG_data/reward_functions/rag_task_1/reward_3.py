import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized offensive maneuver rewards during differing game phases."""

    def __init__(self, env):
        super().__init__(env)
        self.active_modes = {
            0: 1.0,  # Normal play
            1: 0.2,  # Kickoff
            2: 0.1,  # GoalKick
            5: 0.3   # Throw-in
        }
        # Tracking previous offensive states for bonus calculation
        self.previous_ball_position_x = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.previous_ball_position_x = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        # Initialize components for detailed reward explanation
        components = {
            "base_score_reward": reward.copy(),
            "offensive_bonus": 0.0,
            "game_mode_bonus": 0.0
        }
        if observation is None:
            return reward, components
        
        ball_position_x = observation[0]['ball'][0]  # Only considering position on x-axis
        game_mode = observation[0]['game_mode']
        
        # Calculate offensive bonus if the ball is moved towards opponent's goal more aggressively
        if ball_position_x > self.previous_ball_position_x:
            components["offensive_bonus"] = (ball_position_x - self.previous_ball_position_x) * 0.1
        
        # Additional rewards based on game mode switch tactics
        components["game_mode_bonus"] = self.active_modes.get(game_mode, 0)

        total_reward = np.array(reward) + components["offensive_bonus"] + components["game_mode_bonus"]
        self.previous_ball_position_x = ball_position_x

        return total_reward.tolist(), components

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
