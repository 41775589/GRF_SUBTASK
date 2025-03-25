import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards precise high kicks."""

    def __init__(self, env):
        super().__init__(env)
        self.reset()

    def reset(self):
        """Reset the high kick positions tracker."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store checkpoint wrapper state in a picklable format."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore checkpoint wrapper state from a picklable format."""
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Reward agents for executing skillful high kicks (passes) under various conditions,
        with additional reward components for trajectory control and power assessment.
        """
        observation = self.env.unwrapped.observation()
        new_rewards = reward.copy()
        reward_components = {"base_score_reward": reward, "high_kick_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, reward_components

        for i in range(len(reward)):
            player_obs = observation[i]
            
            # Check if the player performed a high pass using 'ball_direction' and 'ball_rotation' data
            if player_obs['ball_owned_team'] == 0 and player_obs['ball'][2] > 0.15:
                # Assuming ball[2] represents height and a height greater than 0.15 is considered high
                ball_direction = player_obs['ball_direction']
                ball_speed_vertical = ball_direction[2]
                
                # High vertical velocity indicates a strong kick aimed upwards
                if ball_speed_vertical > 0.1:
                    # Award players for effective control (higher vertical velocity of the ball)
                    reward_component = 0.2 * ball_speed_vertical
                    new_rewards[i] += reward_component
                    reward_components["high_kick_reward"][i] = reward_component
                
                # Additional reward if the ball is moving towards an advantageous position
                if player_obs['ball'][0] > 0.8:  # more towards the opponent's goal
                    new_rewards[i] += 0.1
                    reward_components["high_kick_reward"][i] += 0.1

        return new_rewards, reward_components

    def step(self, action):
        """Perform a step in the environment and augment the reward using reward function."""
        observation, reward, done, info = self.env.step(action)
        new_reward, reward_components = self.reward(reward)
        info['final_reward'] = sum(new_reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(value)
        return observation, new_reward, done, info
