import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that focuses on defense and midfield management.
    It promotes strategic positioning and effective ball control between defense and midfield.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.midfield_control = 0.1
        self.defensive_positioning = 0.2

    def reset(self):
        # Reset the environment
        return self.env.reset()

    def get_state(self, to_pickle):
        # Get the state from the environment to save it
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Set the state of the environment from the saved state
        return self.env.set_state(state)

    def reward(self, reward):
        # Modify the reward based on defense and midfield control effectiveness
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward).copy(),
                      "midfield_control": np.zeros_like(reward),
                      "defensive_positioning": np.zeros_like(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward midfield control (e.g., players in the midfield with the ball)
            if abs(o['ball'][0]) < 0.5 and o['ball_owned_team'] == 0:
                components["midfield_control"][rew_index] = self.midfield_control
                reward[rew_index] += self.midfield_control

            # Reward defensive positioning (e.g., players closer to own goal preventing attacks)
            if o['right_team'][o['active']][0] < -0.7 and o['ball_owned_team'] != 0:
                components["defensive_positioning"][rew_index] = self.defensive_positioning
                reward[rew_index] += self.defensive_positioning

        return reward, components

    def step(self, action):
        # Overriding the step method to include reward modification
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
