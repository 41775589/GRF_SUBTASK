import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on defensive actions and positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.interception_reward = 0.5
        self.positioning_reward = 0.1
        self.max_y_distance = 0.42  # half of the field width
        self.goal_line_x = 1  # x-coordinate of the goal line
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and the wrapper states."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Return the state of the environment along with the wrapper's state."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment and the wrapper from the given state."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Reward defensive play by reinforcing interceptions and good positioning."""
        observation = self.env.unwrapped.observation()
        
        # The observation is expected to have a list for each agent
        if observation is None or not observation:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_reward = reward[rew_index]
           
            # Reward for intercepting the ball close to own goal
            if o['ball_owned_team'] == 0 and o['ball'][0] < -0.8:
                components["interception_reward"][rew_index] = self.interception_reward
                current_reward += self.interception_reward
            
            # Reward for positioning - closer to the goal line when not possessing the ball
            if o['ball_owned_team'] != 1 and abs(o['ball'][0] - self.goal_line_x) < 0.1:
                dy_distance = abs(o['ball'][1]) 
                positioning_score = (self.max_y_distance - dy_distance) / self.max_y_distance
                additional_reward = self.positioning_reward * positioning_score
                components["positioning_reward"][rew_index] = additional_reward
                current_reward += additional_reward
            
            reward[rew_index] = current_reward
            
        return reward, components

    def step(self, action):
        """Process environment step with the wrapper's reward."""
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
