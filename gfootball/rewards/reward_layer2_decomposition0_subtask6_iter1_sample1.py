import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for performing sliding tackles under high-pressure situations."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Setting a default value for the situation intensity coefficient, which can be adjusted as per environment dynamics
        self.sliding_tackle_intensity_coefficient = 10

    def reset(self):
        """Reset the environment and related variables."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize the current state of this wrapper, including the internal state."""
        to_pickle['CheckpointRewardWrapper'] = {'intensity_coefficient': self.sliding_tackle_intensity_coefficient}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize and apply the state of this wrapper."""
        from_pickle = self.env.set_state(state)
        self.sliding_tackle_intensity_coefficient = from_pickle['CheckpointRewardWrapper']['intensity_coefficient']
        return from_pickle

    def reward(self, reward):
        """Calculate the reward by assessing the quality of the sliding tackle."""
        observation = self.env.unwrapped.observation()
        
        components = {
            "base_score_reward": np.array(reward, copy=True),
            "sliding_tackle_reward": np.zeros_like(reward)
        }

        # To get the best chance of realistic simulation data from the environment, hypothetical function names are used.
        if 'game_mode' in observation and 'ball_owned_team' in observation:
            # Assuming '2' represents a high-pressure defending situation.
            if observation['game_mode'] == 2 and observation['ball_owned_team'] == 0:  # 0 indicating the controlled team
                # Checking the 'sliding' action occurred and was close to the ball (imaginary condition)
                if self.env.last_action == 'sliding' and np.linalg.norm(observation['ball'] - observation['position']) < 0.1:
                    components['sliding_tackle_reward'] += self.sliding_tackle_intensity_coefficient
        
        total_reward = components['base_score_reward'] + components['sliding_tackle_reward']
        return total_reward, components

    def step(self, action):
        """Processing a step with the added reward function."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Update the info dictionary with the reward breakdown.
        info.update({f"component_{key}": value for key, value in components.items()})
        info['final_reward'] = sum(reward)
        
        return observation, reward, done, info
