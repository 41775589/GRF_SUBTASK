import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward specifically tailored for mastering defensive maneuvers, focusing on sliding tackles."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Parameters to define the proximity to the ball where sliding tackles are crucial
        self.sliding_tackle_proximity_threshold = 0.1  # Distance threshold to consider relevance of defensive action

    def reset(self):
        """Reset environment and reward settings."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Saves the current state."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Sets the state of the environment."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Custom reward function focused on improving the effectiveness of sliding tackles."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sliding_tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball']
            player_position = o['right_team'][o['active']]

            # Calculate the distance to the ball
            ball_distance = np.linalg.norm(np.array(ball_position[:2]) - np.array(player_position[:2]))
            
            # Reward sliding tackles when close to ball and in relevant game moments (like defending a lead)
            if ball_distance < self.sliding_tackle_proximity_threshold and o['game_mode'] == 0:  # Normal gameplay
                if any(o['sticky_actions'][9]):  # Check if sliding action is performed
                    components["sliding_tackle_reward"][rew_index] += 1.0  # Reward sliding tackles close to the ball

            # Reward structure: Base game score + tailored sliding tackle rewards
            reward[rew_index] += components["base_score_reward"][rew_index] + components["sliding_tackle_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Step the environment by applying the action, modifying the reward, and returning new state and info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
