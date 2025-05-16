import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that customizes the reward function for mastering 'Stop-Moving' strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset sticky action counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save additional state if necessary."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore saved state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Customize reward to enforce stopping at optimal positions and intercepting passes."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "positioning_reward": [0.0]
        }
        
        if observation is None or len(observation) == 0:
            return reward, components
        
        player_pos = observation[0]['left_team'][observation[0]['active']]
        ball_pos = observation[0]['ball'][:2]
        stop_move_action = observation[0]['sticky_actions'][7]  # Index for 'stop_moving'

        # Calculate the distance to the ball.
        distance_to_ball = np.linalg.norm(ball_pos - player_pos)
        
        # Encourage players to 'Stop-Moving' close to the ball to intercept passes.
        if stop_move_action == 1:
            # Check if within an optimal distance to potentially intercept a pass
            if distance_to_ball < 0.1:
                components['positioning_reward'][0] = 1.0
            else:
                components['positioning_reward'][0] = -0.5

        # Compute the final adjusted reward
        adjusted_reward = reward.copy()
        adjusted_reward[0] += components['positioning_reward'][0]

        return adjusted_reward, components
    
    def step(self, action):
        """Executes a step in the environment."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
