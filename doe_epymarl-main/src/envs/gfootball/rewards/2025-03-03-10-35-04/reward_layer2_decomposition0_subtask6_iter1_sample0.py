import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward specifically designed for mastering Short Passes under pressure."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Track whether the ball was correctly passed under pressure
        self._ball_pass_under_pressure = False

    def reset(self):
        """Reset tracking status for new episodes."""
        self._ball_pass_under_pressure = False
        return self.env.reset()

    def reward(self, reward):
        """Custom reward function that focuses on short passes under defensive pressure."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'pressure_pass_reward': [0.0]}

        if observation is None:
            return reward, components

        # Simplified example: check if the ball is under control and manipulated under pressure
        o = observation[0]  # Since we only have one agent for this subtask
        if o['ball_owned_team'] == 0 and o['left_team_active'][o['active']]:
            # Assuming ball is under control of active agent's team (team 0 - left team)
            ball_owned_player = o['ball_owned_player']
            opponent_dist = np.min(np.linalg.norm(o['right_team'] - o['left_team'][ball_owned_player], axis=1))
            passing_action = o['sticky_actions'][6]  # Assuming index 6 is 'Short Pass'

            # Define pressure threshold and reward for successful pressured pass
            pressure_threshold = 0.2  # Example threshold value
            if opponent_dist < pressure_threshold and passing_action:
                if not self._ball_pass_under_pressure:
                    components['pressure_pass_reward'][0] = 5.0  # Reward for successful short pass under pressure
                    self._ball_pass_under_pressure = True
    
        # Calculate final reward
        reward[0] = reward[0] + components['pressure_pass_reward'][0]
        return reward, components

    def step(self, action):
        """Perform environment step and augment reward.”"""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = reward[0]
        info.update({f"component_{key}": value[0] for key, value in components.items()})
        return observation, reward, done, info
