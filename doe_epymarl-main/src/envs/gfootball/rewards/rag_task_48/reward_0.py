import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to award extra points for successful high passes from midfield leading to scoring opportunities."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.moving_average_position = 0
        self.previous_ball_position = [0, 0]
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_reward_coefficient = 1.0

    def reset(self):
        """Reset the environment and reward parameters for a new game."""
        self.moving_average_position = 0
        self.previous_ball_position = [0, 0]
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize the current state of the wrapper for checkpoints."""
        to_pickle['moving_average_position'] = self.moving_average_position
        to_pickle['previous_ball_position'] = self.previous_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize the state of the wrapper for checkpoints."""
        from_pickle = self.env.set_state(state)
        self.moving_average_position = from_pickle['moving_average_position']
        self.previous_ball_position = from_pickle['previous_ball_position']
        return from_pickle

    def reward(self, reward):
        """Defines a reward function for making successful high passes from midfield."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            current_ball_position = o['ball'][:2]  # Get current x, y position of the ball
            ball_ownership = o['ball_owned_team']  # Which team owns the ball
            
            # Check for a high pass (z position of the ball) from midfield to opponent's half
            if ball_ownership == 0 and self.previous_ball_position[1] < 0.2 < current_ball_position[1] and o['ball'][2] > 0.10:
                # Approximate midfield region could be y-coordinate around 0.2 to -0.2
                if abs(self.previous_ball_position[0]) < 0.3:
                    components["high_pass_reward"][rew_index] = self.high_pass_reward_coefficient
                    reward[rew_index] += components["high_pass_reward"][rew_index]
            
            self.previous_ball_position = current_ball_position  # Update last ball position

        return reward, components

    def step(self, action):
        """Execute a step and add extra reward components."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Update sticky actions
        for idx, agent_obs in enumerate(obs):
            sticky_actions = agent_obs['sticky_actions']
            self.sticky_actions_counter += sticky_actions
            for i, action in enumerate(sticky_actions):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
